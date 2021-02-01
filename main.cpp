#include <iostream>
#include <fstream>
#include <sstream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include "include/ocv_versioninfo.h"
#include "include/FeLibs.h"
#include "include/Macros.h"

#include <thread>

int main( int argc, char* argv[] )
{
    // to unpack cards from 'filename' to a folder for ranks and one for seeds
    // std::string filename = "../src/cards.jpg";
    // extract_cards(filename);

    // read card information and fill global variables 'ranks' and 'seeds'
    read_cards();
    // initialize global variable 'av_cards' to true since all cards are available
    memset(av_cards, true, 52*sizeof(bool));

    int nr_rounds_empty=0;

    std::string filename = "src/test_video_2.mp4";
    cv::VideoCapture cap( filename );
    if( !cap.isOpened() ){
        std::cout << "Error loading video. Check path and try again." << std::endl;
        std::cout << filename << std::endl;
        return -1;
    }

    cv::Mat frame;
    cv::Mat large_frame;
    // Skip some frames
    // cap.set(cv::CAP_PROP_POS_MSEC, 153000);

    /**
     * history: contains detected ids for all cards on the table, for 'h_size' frames,
     *      history[i] represents a card, and the next detect id for that card is
     *      going to be added on position 'history_pointer[i]'.
     * h_ids[i]: contains the id of card 'i' after it appears for a certain number of frames,
     *      making it a confirmation for that id; -1 if not verified yet
     * on_table: contains the ids detected on last analysed frame, needed to check that the
     *      inserted ids on history keep showing up, so for each frame in which they are not
     *      present, a -1 on their history is added,
     *      until they get cleared (when entry history[i] contains only -1s)
     * positions: contains the position on table of all cards in 'history'
     *
     * history[i], history_pointer[i], h_ids[i], on_table[i]
     * and positions[i] all refer to the same card i.
     */
    std::vector<std::vector<int>> history;
    std::vector<int> history_pointer;
    std::vector<int> h_ids; // verified ids
    std::vector<int> remove_cards;
    std::vector<int> on_table;
    std::vector<cv::RotatedRect> positions;

    // for debugging
    std::vector<double> confidence; // similarity of detected object on actual frame
    std::vector<int> debug_ids; // detected id of objects on actual frame

    //cv::namedWindow("frame", cv::WINDOW_AUTOSIZE);

    bool play=true;
    bool quit=false;
    bool more_info=false;
    bool single_frame=true;
    std::string debug_info = "Pause";
    cv::Mat debug_frame;

    int frame_nr=0;
    int total_rank=0;
    float prob_over=0;
    float prob_below=0;
    float prob_bj=0;

    // One frame per time
    while(!quit) {
        if (play) {
            if(single_frame){
                play=false;
                single_frame=false;
            }
            cap >> large_frame; // 2-3 ms.
            if (large_frame.empty()) {
                std::cout << "Empty Frame\n";
                return 0;
            }
            // 1-2 ms.
            cv::resize(large_frame, frame, cv::Size(), 0.5, 0.5, cv::INTER_NEAREST);
            // 7-8 ms. to find boundaries
            //  % 5-6ms. to prepare img
            //  % 2-3ms. to effectively detect boundaries
            cv::Mat gray;
            cv::cvtColor(frame, gray, cv::COLOR_RGB2GRAY); // 0-1 ms.
            int otsu_level = otsu_threshold(gray, 256); // 1 ms.

            apply_threshold_gray(&gray, otsu_level); // 0 ms.
            cv::Mat pad_gray;
            pad_gray = add_padding(gray, 1); // 1 ms.

            cv::Mat hole_gray;
            hole_gray = fill_holes(pad_gray); // 0-1 ms.
            std::vector <std::vector<cv::Point2i>> bond;
            // if otsu_level is suspiciously low, it is better to blur the image
            // since it is going to contain a lot of noisy points (which can slow down
            // significantly the detection of boundaries)
            if (otsu_level < 50) {
                cv::medianBlur(hole_gray, hole_gray, 5);
            }

            bond = moore_boundary(hole_gray, 100); // 2-3 ms.

            // "remove padding"
            for (int i = 0; i < bond.size(); ++i) {
                for (int k = 0; k < bond[i].size(); k++) {
                    --bond[i][k].x;
                    --bond[i][k].y;
                }
            }

            // detect objects on frame, based on extracted boundaries
            std::vector <std::vector<cv::Point2f>> corners;
            detect_objects(frame, bond, corners); // 0ms
            bond.clear();

            // with 3 cards: 20-35 ms.
            for (int ob_idx = 0; ob_idx < corners.size(); ++ob_idx) {
                double sim = INT_MAX;

                // 3-12 ms.
                int id = detect_card_id(frame, corners[ob_idx], &sim);

                confidence.push_back(sim);
                if (sim<min_sim && id!=-1) {
                    int h_entry = -1;
                    // Check whether id was already detected on table
                    for (int e = 0; e < h_ids.size(); ++e) {
                        if (h_ids[e] == id) {
                            h_entry = e;
                            on_table[e]=-1;
                        }
                    }

                    // retrieve Rotated Rectangle corresponding to the position of the detected card
                    cv::RotatedRect card_rect(corners[ob_idx][0], corners[ob_idx][1], corners[ob_idx][2]);
                    if (h_entry == -1) { // id not detected on last_tab, check position
                        // check whether it clashes with other cards
                        // 0 ms.
                        for (int e = 0; e < positions.size() && h_entry < 0; ++e) {
                            float int_prc = intersection_percentage(positions[e], card_rect, frame.size());
                            // if there is partial or full intersection with this card in position e
                            // then add this id to history[e]
                            if (int_prc>0.8) {
                                h_entry = e;
                                history[e][history_pointer[e]] = id;
                                history_pointer[e] = (history_pointer[e] + 1) % h_size;
                                // update position on table
                                positions[e] = card_rect;
                            }
                        }

                        // if we didn't find an intersection in position, then it is a new card on table
                        if (h_entry == -1) {
                            // add a new column in history for this new detected card,
                            // add a corresponding pointer in history_pointer
                            history.push_back(std::vector<int>(h_size, -1));
                            history_pointer.push_back(0);
                            h_ids.push_back(-1);
                            history.back()[history_pointer.back()] = id;
                            history_pointer.back() = ((history_pointer.back()) + 1 % h_size);
                            // save the position on table of this new card
                            positions.push_back(card_rect);
                        }

                    } else { // id detected on last_table [h_entry=position in history]
                        history[h_entry][history_pointer[h_entry]] = id;
                        history_pointer[h_entry] = (history_pointer[h_entry] + 1) % h_size;
                        // update position on table
                        positions[h_entry] = card_rect;
                    }

                } // end of if(sim<min_sim && id!=-1)
            }
            // at this point we checked all detected objects and updated history

            // for cards of last table not in this frame, add a -1 to their history
            for (int e = 0; e < on_table.size(); ++e) {
                if (on_table[e] != -1) {
                    history[e][history_pointer[e]] = -1;
                    history_pointer[e] = (history_pointer[e] + 1) % h_size;
                    // the following commented code makes those cards which are
                    // currently occluded but verified visible on debugging phase.
                    // bit ugly since the position becomes obsolete after a while
                    /**if(h_ids[e]!=-1){
                        cv::Point2f rect_points[4];
                        positions[e].points(rect_points);
                        std::vector<cv::Point2f> temp(4);
                        for(int e=0; e<4; ++e) temp[e] = rect_points[e];
                        corners.push_back(temp);
                        confidence.push_back(1);
                    }**/
                }
            }

            on_table.clear();
            std::vector<int> to_remove;
            bool isEmpty = true;
            // determine cards on table - each column of 'history' corresponds to a card
            for (int col = 0; col < history.size(); ++col) { // 0 ms.
                int most_occurring_id = history[col][0];
                int nr_occurrences = 1, tmp_occurrences;
                int tmp = 0;

                for (int i = 0; i < history[col].size() - 1; ++i) {
                    tmp = history[col][i];
                    tmp_occurrences = 0;
                    if (tmp != -1) {
                        for (int j = 0; j < history[col].size(); ++j) {
                            if (tmp == history[col][j]) ++tmp_occurrences;
                        }
                        if (tmp_occurrences > nr_occurrences) {
                            nr_occurrences = tmp_occurrences;
                            most_occurring_id = tmp;
                        }
                    }
                }

                on_table.push_back(most_occurring_id);
                // add the most occurring element in 'on_table'
                if (most_occurring_id != -1) {
                    if (nr_occurrences > verify){
                        h_ids[col] = most_occurring_id;
                    }
                    isEmpty=false;
                } else {
                    // if this column is made of only -1, remove only if h_ids is -1
                    if(h_ids[col]==-1) {
                        to_remove.push_back(col);
                    }
                }
            }

            // remove empty columns in 'to_remove' in a safe way:
            for (int e = to_remove.size() - 1; e > -1; --e) {
                int idx_remove = to_remove[e];
                history.erase(history.begin() + idx_remove);
                history_pointer.erase(history_pointer.begin() + idx_remove);
                h_ids.erase(h_ids.begin() + idx_remove);
                positions.erase(positions.begin() + idx_remove);
                on_table.erase(on_table.begin() + idx_remove);
            }

            // calculate probabilities based on 'h_ids' which is more stable
            calculate_probabilities(total_rank, prob_below, prob_bj, prob_over, h_ids);

            if (debug) { // 7 ms. (mostly because of cv::imshow)
                debug_ids.clear();
                // determine debug_ids - 0 ms.
                for (int ob_idx = 0; ob_idx < corners.size(); ++ob_idx) {
                    int h_entry = -1;
                    // retrieve Rotated Rectangle of object in corners[ob_idx]
                    cv::RotatedRect card_rect(corners[ob_idx][0], corners[ob_idx][1], corners[ob_idx][2]);
                    for (int e = 0; e < positions.size() && h_entry < 0; ++e) {
                        float int_prc = intersection_percentage(positions[e], card_rect, frame.size());

                        // if there is enough intersection with this card in position e
                        if (int_prc > 0.95) {
                            h_entry = e;
                            debug_ids.push_back(h_ids[h_entry]);
                        }
                    }
                    if (h_entry == -1) debug_ids.push_back(-1);
                }

                debug_frame = frame.clone();
                // 1 ms.
                show_info(frame, frame_nr, confidence, corners, debug_ids, h_ids, debug_info, more_info,
                        total_rank, prob_below, prob_bj, prob_over);
                // 6 ms.
                cv::imshow("BlackJack", frame);
                char key = cv::waitKey(5);

                if(key == 'p') {
                    play = false;
                    debug_info = "Pause";
                    show_info(debug_frame, frame_nr, confidence, corners, debug_ids, h_ids, debug_info, more_info,
                              total_rank, prob_below, prob_bj, prob_over);
                    cv::imshow("BlackJack", debug_frame);
                }
                if(key == 'i'){
                    more_info=!more_info;
                    show_info(debug_frame, frame_nr, confidence, corners, debug_ids, h_ids, debug_info, more_info,
                              total_rank, prob_below, prob_bj, prob_over);
                    cv::imshow("BlackJack", debug_frame);
                }
                if(key == 'q') quit=true;
                if(key == 'o') {
                    single_frame=true;
                    play=true;
                    debug_info = "Pause";
                }
                debug_frame.release();
            }

            // if board is empty remove cards
            if (isEmpty && h_ids.size()>0) {
                std::cout << "Removing:\t";
                for (int e = h_ids.size()-1; e > -1; --e) {
                    if(h_ids[e]!=-1) {
                        std::cout << id_to_text(h_ids[e]) << " ";
                        av_cards[h_ids[e]] = false;
                    }
                    history.erase(history.begin() + e);
                    history_pointer.erase(history_pointer.begin() + e);
                    h_ids.erase(h_ids.begin() + e);
                    positions.erase(positions.begin() + e);
                }
                std::cout << "\n";
                nr_rounds_empty = 0;
                on_table.clear();
            }

            // ------------------------- //
            confidence.clear();
            large_frame.release();
            gray.release();
            pad_gray.release();
            hole_gray.release();
            corners.clear();
            frame.release();
            // ------------------------- //

            ++frame_nr;

        } // end of if(play)
        else{ // if (play==false) wait for key_press event
            char key = cv::waitKey(0);
            if(key == 'p') {
                play = true;
                debug_info = "Play";
            }
            if(key == 'q') quit=true;
            if(key == 'i'){
                more_info=!more_info;
                single_frame=true;
                play=true;
            }
            if(key == 'o'){
                single_frame=true;
                play=true;
            }
        }
    } // end of while(!quit)

    std::cout << "Program quitting after key 'q' pressed\n";

    if(!history.empty()) {
        history.clear();
        history_pointer.clear();
        h_ids.clear();
        positions.clear();
    }
    ranks.clear();
    seeds.clear();
    debug_ids.clear();
    on_table.clear();

    cv::destroyWindow("BlackJack");

    // cap.release(); automatically called
    return 0;
}



#ifndef MACROS_H
#define MACROS_H

#include <iostream>
#include <fstream>
#include <sstream>
#include <float.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include "FeLibs.h"

#define debug true // if debug is false you will see nothing

#define black_bg CV_RGB(33,33,33)
#define red_bg CV_RGB(211,47,47)
#define font_col CV_RGB(255,255,255)
#define green_box CV_RGB(102,187,106)
#define red_box CV_RGB(239,83,80)
#define err_bg CV_RGB(97,97,97)
#define back_bg CV_RGB(42,42,42)
#define light_green CV_RGB(165,214,167)
#define light_red CV_RGB(239,154,154)

#define min_sim 16000 // minimum similarity to consider a card correctly detected
#define h_size 35 // number of frames of history
#define verify 10 // nr of frames to consider identity of a card as verified
bool av_cards[52]; // available cards in game
std::vector<cv::Mat> ranks;
std::vector<cv::Mat> seeds;

/**
 * Extract cards information, that is ranks and seeds,
 * add them in src/ranks/r_name.png and src/seeds/s_name.png where:
 * r_name = {1, 2, ...13}.png respectively for cards Ace, Two ...Ten, Jack(11), Queen(12), King(13)
 * s_name = {0, 1, 2, 3}.png respectively for Spades, Clubs, Diamonds and Hearts
 * Images are saved in grayscale, where pixel values are at 0 or 255.
 *
 * First the region corresponding to a single card is extracted at a time,
 * then the upper corner of the card is considered since info relies there.
 * Once the region is extracted, moore boundary algorithm is applied to retrieve only
 * the rectangle of area bounding the rank or the seed.
 */
void extract_cards(std::string filename){
    cv::Mat I = cv::imread( filename );
    if( I.data == NULL ){
        std::cout << "Error loading cards. Check path and try again." << std::endl;
        return;
    }

    int i=0;
    for(int x=0; x<I.cols; x=x+I.cols/4){
        for(int y=0; y<I.rows; y=y+I.rows/13){
            ++i;
            if(i==53) break;
            if(i<14 || i%13==1){
                // get the proportion of the img corresponding to card i
                cv::Rect card_rect(x,y,I.cols/4, I.rows/13);
                cv::Mat card_region = I(card_rect);
                cv::Mat ul_corner; // upper-left corner

                // extract approximately only the part of the card where rank and seed appears
                cv::Rect ul_rect(cv::Point2i(20, 20), cv::Point2i(x, 180));
                ul_corner = card_region(ul_rect);

                // convert to gray and apply threshold
                cv::cvtColor(ul_corner, ul_corner, cv::COLOR_RGB2GRAY);
                int otsu_level = otsu_threshold(ul_corner, 256);
                apply_threshold_gray(&ul_corner, otsu_level);

                // remove noise / imperfections
                cv::medianBlur(ul_corner, ul_corner, 3);
                apply_threshold_gray(&ul_corner, otsu_level);

                // invert pixels in order to feed it to moore_boundary
                cv::Mat inv_up;
                bitwise_not(ul_corner, inv_up);
                // add padding, needed for moore_boundary
                inv_up = add_padding(inv_up, 1);
                // add a line to push moore_boundary to detect '10' as a single object
                cv::line(inv_up, cv::Point2i(3, 50), cv::Point2i(inv_up.cols-3, 50), cv::Scalar(255,255,255), 1);
                // fill holes, needed for moore_boundary
                inv_up = fill_holes(inv_up);

                std::vector <std::vector<cv::Point2i>> bond = moore_boundary(inv_up, 100);
                for (int e = 0; e < bond.size(); ++e) {
                    for (int q = 0; q < bond[e].size(); q++) {
                        --bond[e][q].x;
                        --bond[e][q].y;
                    }
                }

                // save ranks
                if(i<14 && i!=10) {
                    std::string path = "src/ranks/" + std::to_string(i) + ".png";
                    ul_corner(cv::boundingRect(bond[0]));

                    cv::Mat to_bound;
                    bitwise_not(ul_corner, to_bound);
                    to_bound = add_padding(to_bound, 1);
                    to_bound = fill_holes(to_bound);

                    std::vector <std::vector<cv::Point2i>> boundaries = moore_boundary(to_bound, 10);
                    for (int e = 0; e < boundaries.size(); ++e) {
                        for (int q = 0; q < boundaries[e].size(); q++) {
                            --boundaries[e][q].x;
                            --boundaries[e][q].y;
                        }
                    }
                    cv::imwrite(path, ul_corner(cv::boundingRect(boundaries[0])));
                    to_bound.release();
                    boundaries.clear();
                }
                if(i==10){
                    std::string path = "src/ranks/" + std::to_string(i) + ".png";
                    cv::imwrite(path, ul_corner(cv::boundingRect(bond[0])));
                }

                // save seeds
                if(i%13==1) {
                    std::string path = "src/seeds/" + std::to_string(i/13) + ".png";
                    cv::Mat bl = ul_corner(cv::boundingRect(bond[1]));
                    cv::imwrite(path, bl);
                    bl.release();
                }
                card_region.release();
                ul_corner.release();
                bond.clear();
                inv_up.release();
            }
        }

    }
}

/**
 * Read card ranks and seeds from path /src/ranks/.. and /src/seeds/..
 * and load them into global variables 'ranks' and 'seeds'
 */
void read_cards(){
    std::string path = "src/ranks/";
    for(int i=1; i<14; ++i){
        // read image in grayscale
        cv::Mat I = cv::imread(path+std::to_string(i)+".png", cv::IMREAD_GRAYSCALE);
        ranks.push_back(I);
        I.release();
    }
    path = "src/seeds/";
    for(int i=0; i<4; ++i){
        // read image in grayscale
        cv::Mat I = cv::imread(path+std::to_string(i)+".png", cv::IMREAD_GRAYSCALE);
        seeds.push_back(I);
        I.release();
    }
}

/**
 * Evaluate the amount of red pixels inside the image.
 * Consider only pixels where in grayscale they have a value > 140.
 *
 * @param input image to be evaluated
 * @return percentage of red (values > 0.1 represent red cards)
 */
float redness(cv::Mat &input){
    float red_pts=0, rgb=0;
    cv::Mat gray_in;
    cv::cvtColor(input, gray_in, cv::COLOR_RGB2GRAY);

    apply_threshold_gray(&gray_in, 140);

    int gray=0,b,g,r;
    for(int x=0; x<gray_in.rows; ++x){
        for(int y=0; y<gray_in.cols; ++y){
            gray = gray_in.at<int>(x,y);
            b = input.at<cv::Vec3b>(x,y)[0];
            g = input.at<cv::Vec3b>(x,y)[1];
            r = input.at<cv::Vec3b>(x,y)[2];
            if(gray > 10){
                // if it is the region of rgb space visually perceivable as red
                if(r>b && r>g && r-b>80 && r-g>80) ++red_pts;
                ++rgb;
            }
        }
    }
    gray_in.release();

    if(rgb==0) ++rgb;
    return red_pts/rgb;
}

/**
 * Rotate circularly the vector of points by one position
 * @param pts vector of points to be rotated
 */
void shift_around(std::vector<cv::Point2f> &pts){
    cv::Point2f temp = pts[0];
    for(int e=0; e<pts.size()-1; ++e){
        pts[e] = pts[e+1];
    }
    pts[pts.size()-1] = temp;
}

/**
 * improvements done in similarity calculation between 2 cards:
 *      accessing images with unsigned char* pointers;      - 1-2 ms.
 *      having images already in gray_scale and thresholded - 1-2 ms.
 */

/**
 * Detect the seed identity in seed_rect.
 * Mean Squarred Error is used to measure similarity between images.
 *
 * @param seed_rect Image where seed information relies
 * @param sim similarity to the detected seed is going to be placed here
 * @param red indicates whether the card is red or not (redness>0.1)
 * @return {0, 1, 2, 3}: the index of global variable 'seeds' entry
 *         corresponding to the detected seed
 */
int detect_seed(cv::Mat &seed_rect, double *sim, bool red){
    int detected_seed=-1;
    double best_sim=INT_MAX;
    double temp_sim;

    int initial=0;
    if(red) initial = 2;

    for(int s_idx=initial; s_idx<seeds.size(); ++s_idx){
        double mse=0;
        int row=0;
        int n=0;

        while(row < seed_rect.rows && row < seeds[s_idx].rows){
            unsigned char *bob_ptr = seed_rect.ptr(row);
            unsigned char *bob_end = bob_ptr + seed_rect.cols;
            unsigned char *tom_ptr = seeds[s_idx].ptr(row);
            unsigned char *tom_end = tom_ptr + seeds[s_idx].cols;

            while (bob_ptr != bob_end && tom_ptr != tom_end) {
                if( (*bob_ptr)==0 || (*tom_ptr)==0){
                    mse += std::pow((*bob_ptr)-(*tom_ptr), 2);
                }
                ++bob_ptr;
                ++tom_ptr;
                ++n;
            }
            ++row;
        }
        temp_sim = mse/n;

        if(temp_sim < best_sim){
            best_sim = temp_sim;
            detected_seed = s_idx;
        }
    }
    *sim = best_sim;
    return detected_seed;
}

// Improvement idea: use async threads for different shifts
// Improvement done: slide in vertical and horizontal, big accuracy improvement
/**
 * Detect the rank identity in rank_rect.
 * Reference images in 'ranks' global variable should be smaller in size (at least horizontally),
 * for which reason it is slided horizontally over the input image and
 * the best result is considered for each rank.
 *
 * Mean Squarred Error is used to measure similarity between images.
 *
 * @param rank_rect image containing the rank
 * @param sim similarity to the detected rank is going to be placed here
 * @return {0-13}: the index of global variable 'ranks' entry
 *         corresponding to the detected rank
 */
int detect_rank_slide(cv::Mat &rank_rect, double *sim){
    int detected_rank=-1;
    double best_sim = INT_MAX; // most similar among all ranks
    double shifts_sim; // most similar among all shifts for actual rank
    double mse, temp_sim;
    int row, n, rows, cols;

    // compare to all possible ranks (stored in global variable 'ranks')
    for (int r_idx = 0; r_idx < ranks.size(); ++r_idx) {
        // rename the two images to be compared, for clarity
        cv::Mat img1, img2;
        img1 = rank_rect;
        img2 = ranks[r_idx];

        rows = std::max(img1.rows, img2.rows);
        cols = std::max(img1.cols, img2.cols);

        mse=0;
        temp_sim = INT_MAX;
        shifts_sim = INT_MAX;

        for (int row_shift = 0; row_shift < rows - img2.rows; row_shift = row_shift + 2) {
            for (int col_shift = 0; col_shift < cols - img2.cols; col_shift = col_shift + 2) {
                mse = 0;
                row = row_shift;
                n = 0;

                while (row < img1.rows && row < img2.rows) {
                    unsigned char *bob_ptr = img1.ptr(row) + col_shift;
                    unsigned char *bob_end = bob_ptr + img1.cols - col_shift;
                    unsigned char *tom_ptr = img2.ptr(row);
                    unsigned char *tom_end = tom_ptr + img2.cols;

                    while (bob_ptr != bob_end && tom_ptr != tom_end) {
                        if ((*bob_ptr) == 0 || (*tom_ptr) == 0) {
                            mse += std::pow((*bob_ptr) - (*tom_ptr), 2);
                        }
                        ++bob_ptr;
                        ++tom_ptr;
                        ++n;
                    }
                    ++row;
                }

                temp_sim = mse / n;

                if (shifts_sim > temp_sim) {
                    shifts_sim = temp_sim;
                }
            }
        }
        if (shifts_sim < best_sim) {
            best_sim = shifts_sim;
            detected_rank = r_idx;
        }
        img1.release();
        img2.release();
    }
    *sim = best_sim;
    return detected_rank;
}

/**
 * Given the part of the card where rank and seed information relies,
 * extract the rank and seed detected on it.
 * - converts to grayscale
 * - removes noisy information by blurring
 * - applies Laplacian sharpening filter
 * - for each detected object,
 *      . if it has the size of a rank, calls detect_rank_slide
 *      . if it has the size of a seed, calls detect_seed
 *
 * @param corner_rect image region where rank and seed info relies
 * @param most_sim similarity to the detected id is going to be placed here
 * @return detected card identity
 */
int extract_info(cv::Mat &corner_rect, double *most_sim){
    int most_idx=-1;

    // convert to gray and apply threshold
    cv::Mat gray_img;
    cv::cvtColor(corner_rect, gray_img, cv::COLOR_RGB2GRAY);
    int otsu_level = otsu_threshold(gray_img, 256);
    apply_threshold_gray(&gray_img, otsu_level);

    // remove noise / imperfections
    cv::medianBlur(gray_img, gray_img, 3);
    apply_threshold_gray(&gray_img, otsu_level);

    // apply Laplacian sharpening filter
    cv::Mat dst_laplace, img_laplaced;
    cv::Laplacian(gray_img, dst_laplace, CV_16S, 3);
    cv::convertScaleAbs(dst_laplace, img_laplaced);

    // add padding in order to execute moore_boundary
    cv::Mat pad_img;
    pad_img = add_padding(img_laplaced, 1);
    // add a line (length:72) to force moore_boundary to detect '10' as a single object
    cv::line(pad_img, cv::Point2i(13, 60), cv::Point2i(pad_img.cols-23, 60), cv::Scalar(255,255,255), 1);
    // fill holes in order to execute moore_boundary
    cv::Mat no_holes_img;
    no_holes_img = fill_holes(pad_img);

    // "remove padding"
    std::vector <std::vector<cv::Point2i>> bond = moore_boundary(no_holes_img, 70);
    for(int i=0; i<bond.size(); ++i){
        for(int j=0; j<bond[i].size(); ++j){
            --bond[i][j].x;
            --bond[i][j].y;
        }
    }

    int detected_rank = -1, detected_seed=-1;
    double sim_rank=0, best_rank=INT_MAX;
    double sim_seed=0, best_seed=INT_MAX;
    for(int b_idx=0; b_idx<bond.size(); ++b_idx){
        cv::Rect tmp(cv::boundingRect(bond[b_idx]));
        cv::Mat rank_rect;
        rank_rect = gray_img(tmp);

        // if it has the size of rank images, detect rank id
        if(std::abs(rank_rect.rows-69) < 20 &&
           (std::abs(rank_rect.cols-70)<20 || std::abs(rank_rect.cols-40)<20)){
            int tmp = detect_rank_slide(rank_rect, &sim_rank);
            if(sim_rank<best_rank) {
                detected_rank = tmp;
                best_rank = sim_rank;
            }
        }
        // if it has the size of seed images, detect seed id
        if(std::abs(rank_rect.rows-42) < 20 && std::abs(rank_rect.cols-47) < 20){
            cv::Mat test_redness;
            test_redness = corner_rect(tmp);
            float red = redness(test_redness);
            int tmp = detect_seed(rank_rect, &sim_seed, red>0.1);
            if(sim_seed<best_seed){
                detected_seed = tmp;
                best_seed = sim_seed;
            }
            test_redness.release();
        }
        rank_rect.release();
    }
    if(detected_rank==-1 || detected_seed==-1){
        *most_sim = INT_MAX;
        detected_rank = -1;
    }else {
        *most_sim = (best_rank + best_seed) / 2;
        detected_rank += 13 * detected_seed;
    }

    gray_img.release();
    dst_laplace.release();
    img_laplaced.release();
    pad_img.release();
    no_holes_img.release();

    return detected_rank;
}

 /**
  * Returns the index of most similar card and its similarity measure.
  * - crops the card to the upper-left corner, where rank and seed info relies
  *      otherwise tries on the bottom-right corner
  *
  * @param img size:[458,640] image region of the card
  * @param most_sim similarity is going to be placed here
  * @return id of the most similar card id
  */
int most_similar(cv::Mat &img, double *most_sim){
    cv::Rect upper_corner(cv::Point2i(20,20), cv::Point2i(100,180));
    cv::Mat up_img;
    up_img = img(upper_corner);

    int detected_rank = extract_info(up_img, most_sim);

    // if the id is not correctly detected or it is not available (removed from game)
    // then try to detect id from bottom right corner, rotated
    if(detected_rank==-1 || !av_cards[detected_rank] || *most_sim>min_sim) {
        *most_sim = INT_MAX;
        // coordinates of the other corner(bottom-right) of the card
        cv::Rect right_corner(cv::Point2i(368,460), cv::Point2i(448,620));
        cv::Mat right_img;
        right_img = img(right_corner);
        cv::Mat rotated_img(right_img.size(), right_img.type());
        // rotate image by 180 degrees
        cv::flip(right_img, rotated_img, -1);
        detected_rank = extract_info(rotated_img, most_sim);
        if(!av_cards[detected_rank]) detected_rank=-1;

        right_img.release();
        rotated_img.release();
    }

    up_img.release();

    return detected_rank;
}

/**
 * Given boundary points of each object, detect the 4 corner points for each of them,
 * and add them to 'corners'
 *
 * @param frame
 * @param bond boundary points detected on frame
 * @param corners vector in which corners are going to be added
 */
void detect_objects(cv::Mat &frame, std::vector<std::vector<cv::Point2i>> &bond,
        std::vector<std::vector<cv::Point2f>> &corners){
    for(int obj=0; obj<bond.size(); ++obj) {
        cv::RotatedRect lerect;
        lerect = cv::minAreaRect(bond[obj]);
        cv::Point2f rect_points[4];
        lerect.points(rect_points);
        std::vector<cv::Point2f> temp(4);
        for(int e=0; e<4; ++e) temp[e] = rect_points[e];
        corners.push_back(temp);
    }
}

/**
 * Transform the id into a readable text representing the id of the card.
 *
 * @param id corresponding id of the card
 * @return text corresponding to the id of the card
 */
std::string id_to_text(int id){
    std::string text = "";
    int number = id%13;
    if(number > 9) {
        if (number == 10) text += "J";
        else {
            if (number == 11) text += "Q";
            else {
                if (number == 12) text += "K";
            }
        }
    }else{
        text += std::to_string(number+1);
    }

    if(id < 13) text += "/S";
    else {
        if(id < 26) text += "/C";
        else {
            if (id < 39) text += "/D";
            else text += "/H";
        }
    }
    return text;
}


/**
 * Given the frame and the 4 corner points of the card, determines its id.
 * - rotates corner points if card is in horizontal position
 * - warps the card into a cv::Mat of size[458,640]
 * - calls ->most_similar (retrieves corner where rank and seed info relies)
 *              ->extract_info (clears image, and crops only on the region where info relies)
 *                  ->detect_rank_slide (determines the most similar rank to the given img)
 *                  ->detect_seed (determines the most similar seed to the given img)
 *
 * @param frame
 * @param corners 4 corner points of the interested area
 * @param sim similarity is going to be placed here
 * @return the id of the detected card
 */
int detect_card_id(cv::Mat &frame, std::vector<cv::Point2f> &corners, double *sim){
    cv::Size out_size(458,640);
    cv::Point2f t_origin = corners[1];

    // rotate corners in case card is detected in horizontal position
    if(std::abs(corners[0].y-corners[1].y)>std::abs(corners[1].x-corners[2].x)
    && corners[0].y > corners[1].y) {
        shift_around(corners);
        t_origin = corners[3];
    }

    // 2 ms. to warp perspective
    std::vector< cv::Point2f > dst_pts;
    dst_pts.push_back( cv::Point2f(0,0) );
    dst_pts.push_back( cv::Point2f(out_size.width,0) );
    dst_pts.push_back( cv::Point2f(out_size.width,out_size.height) );
    dst_pts.push_back( cv::Point2f(0,out_size.height) );

    cv::Mat H = cv::findHomography( corners, dst_pts );
    cv::Mat warped_img;
    cv::warpPerspective( frame, warped_img, H, out_size );

    dst_pts.clear();
    // most expensive call
    int id = most_similar(warped_img, sim);

    warped_img.release();
    return id;
}

/**
 * Returns the amount of shapeA itersected to shapeB
 *
 * @param shapeA
 * @param shapeB
 * @param size size of the frame in which shapes rely
 * @return intersection in range [0, 1] (actually even something more than 1)
 */
float intersection_percentage(cv::RotatedRect shapeA, cv::RotatedRect shapeB, cv::Size size){
    float shape_count=shapeA.size.area(); // returns a float number, which is the reason for return > 1
    float inter_count=0; // pixels of intersection
    float inter_percentage=0;

    if(shape_count<1) return 0;

    std::vector <cv::Point2i> intersection;
    int inter = cv::rotatedRectangleIntersection(shapeA, shapeB, intersection);

    // if there is intersection, draw it in 'im'
    if(inter == cv::INTERSECT_FULL || inter == cv::INTERSECT_PARTIAL) {
        cv::Mat im(size, CV_8U, cv::Scalar(0,0,0));

        // write intersection points in a vector, needed for 'fillConvexPoly'
        cv::Point2i *int_pts;
        int_pts = (cv::Point2i*)malloc(sizeof(cv::Point2i) * intersection.size());
        for (int e = 0; e < intersection.size(); ++e) {
            *(int_pts + e) = intersection[e];
        }

        // draw it with white color, filled
        cv::fillConvexPoly(im, int_pts, intersection.size(), cv::Scalar(255, 255, 255));
        // count nr of pixels just drawn
        inter_count = cv::countNonZero(im);
        // calculate percentage of intersection
        inter_percentage = inter_count/shape_count;

        im.release();
    }

    return inter_percentage;
}


void calculate_probabilities(int &rank, float &prob_below, float &prob_bj, float &prob_over,
        std::vector<int> &h_ids){
    // total rank on table
    rank=0;
    for(int e=0; e<h_ids.size(); ++e){
        if(h_ids[e]>-1){
            int temp=(h_ids[e]%13)+1;
            if(temp>10) temp=10;
            if(temp==1) temp=11;
            rank += temp;
        }
    }
    // prob of over or below 21 on next card
    int missing = 21-rank;
    if(missing>1) {
        int nr_remaining_cards=0;
        prob_below=0;
        prob_over=0;
        prob_bj=0;
        for (int e = 0; e < 52; ++e) {
            if(av_cards[e]){
                ++nr_remaining_cards;
                int temp=(e%13)+1;
                if(temp>10) temp=10;
                if(temp==1) temp=11;

                if(temp>missing) {
                    ++prob_over;
                }
                else{
                    if(temp<missing){
                        ++prob_below;
                    }
                    else{
                        ++prob_bj;
                    };
                }
            }
        }
        prob_below /= nr_remaining_cards;
        prob_bj /= nr_remaining_cards;
        prob_over /= nr_remaining_cards;
    }else{
        prob_over=1;
        prob_below=0;
        prob_bj=0;
    }
}


// for debugging
void show_info(cv::Mat frame, int z,
        std::vector<double> confidence, std::vector<std::vector<cv::Point2f>> corners,
        std::vector<int> ids, std::vector<int> verified_ids, std::string debug_info, bool more_info,
        int rank, float prob_below, float prob_bj, float prob_over){

    int fontface = cv::FONT_HERSHEY_SIMPLEX;
    double scale = 0.5;
    double alpha = 0.7;
    int thickness = 1;
    int baseline = 0;
    cv::Scalar col;

    std::string names[13] = {"1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"};
    std::string seeds[4] = {"S", "C", "D", "H"};

    int nr=0;

    // draw rectangles around cards
    for(int obj=0; obj<corners.size(); ++obj) {
        std::string label;
        if (ids[obj] > -1) {
            col = green_box;
            for (int i = 0; i < 4; i++) cv::line(frame, corners[obj][i],
                                                 corners[obj][(i + 1) % 4], col, 2);
            if (ids[obj] < 26) col = black_bg; // gray/black
            else col = red_bg; // red
            label = id_to_text(ids[obj]);
        } else {
            col = red_box;
            for (int i = 0; i < 4; i++) cv::line(frame, corners[obj][i], corners[obj][(i + 1) % 4], col, 2);
            col = black_bg;
            label = "NA";
        }
        cv::Point2f t_origin = corners[obj][0];
        cv::Point2f tmp = corners[obj][0];
        t_origin.y=0; tmp.y = 0;
        for(cv::Point2f &pt: corners[obj]){
            if(pt.y > t_origin.y){
                t_origin = pt;
            }
        }
        for(cv::Point2f &pt: corners[obj]){
            if(pt.y > tmp.y && pt != t_origin)
                tmp = pt;
        }
        if(t_origin.x>tmp.x)
            t_origin = tmp;

        cv::Size text_size = cv::getTextSize(label, fontface, scale, thickness, &baseline);
        cv::rectangle(frame, t_origin + cv::Point2f(0, baseline),
                      t_origin + cv::Point2f(text_size.width, -text_size.height), col, cv::FILLED);
        if (ids[obj] > -1)
            cv::putText(frame, label, t_origin, fontface, scale, font_col, thickness, 8);
        else
            cv::putText(frame, label, t_origin, fontface, scale, err_bg, thickness, 8);
    }

    // draw background
    cv::Mat overlay;
    frame.copyTo(overlay);
    cv::rectangle(overlay, cv::Point2f(4, frame.rows-1),
            cv::Point2f((24*13)+14, frame.rows-4-(24*4)+8), back_bg, -1);

    cv::rectangle(overlay, cv::Point2f((13*24)+22, frame.rows-1),
            cv::Point2f(frame.cols-4, frame.rows-22), back_bg, -1);

    cv::addWeighted(overlay, alpha, frame, 1-alpha, 0, frame);

    // draw a table with all possible cards on bottom left corner of frame
    cv::Point2f pos(4,frame.rows-4);
    for(std::string &seed: seeds) {
        for(std::string &name: names) {
            col = black_bg;
            for(int &e: verified_ids)
                if(e==nr) col = green_box;
            std::string label = name + seed;
            cv::Size text_size = cv::getTextSize(label, fontface, scale, thickness, &baseline);
            cv::rectangle(frame, pos+cv::Point2f(0, baseline),
                    pos+cv::Point2f(text_size.width, -text_size.height-6),col, 2);
            if(!av_cards[nr]) col = black_bg;
            else col = font_col;
            cv::putText(frame, label, pos, fontface, scale, col, thickness, 8);
            pos.x += 24;
            if(nr%13==9) pos.x += 12; //string "10" is longer
            ++nr;
        }
        pos.x=4;
        pos.y-=24;
    }

    // write frame number on top left corner
    std::string text = "frame: " + std::to_string(z);
    cv::putText(frame, text, cv::Point2i(0,20), fontface, scale, font_col, thickness, 8);
    // write video status (play, pause)
    cv::putText(frame, debug_info, cv::Point2i(0,40), fontface, scale, font_col, thickness, 8);


    // write probability values
    pos.x=4; pos.y=frame.rows-4-(24*4);
    text = "Below: " + std::to_string((int)(prob_below*100)) + "%";
    cv::putText(frame, text, pos, fontface, scale, light_green, thickness, 8);

    pos.y-=16;
    text = "Over:  " + std::to_string((int)(prob_over*100)) + "%";
    cv::putText(frame, text, pos, fontface, scale, light_red, thickness, 8);

    pos.x+=130; pos.y+=16;
    text = "BlackJack: " + std::to_string((int)(prob_bj*100)) + "%";
    cv::putText(frame, text, pos, fontface, scale, light_green, thickness, 8);


    // write legend (press 'p', 'q', 'i', 'o')
    pos.x = 13*24 + 22;
    pos.y = frame.rows-6;
    col = black_bg;

    text = "P: pause";
    cv::Size text_size = cv::getTextSize(text, fontface, scale, thickness, &baseline);
    cv::rectangle(frame, pos+cv::Point2f(-1, baseline),
                  pos+cv::Point2f(text_size.width+1, -text_size.height-6),col, 2);
    pos.x+=1;
    if(debug_info!="Play") text = "P: play";
    cv::putText(frame, text, pos, fontface, scale, cv::Scalar(255,255,255), thickness, 8);
    pos.x+=1;

    text = "Q: quit";
    pos.x += text_size.width+4;
    text_size = cv::getTextSize(text, fontface, scale, thickness, &baseline);
    cv::rectangle(frame, pos+cv::Point2f(-1, baseline),
                  pos+cv::Point2f(text_size.width+1, -text_size.height-6),col, 2);
    pos.x+=1;
    cv::putText(frame, text, pos, fontface, scale, cv::Scalar(255,255,255), thickness, 8);
    pos.x+=1;

    text = "I: info";
    pos.x += text_size.width+4;
    text_size = cv::getTextSize(text, fontface, scale, thickness, &baseline);
    cv::rectangle(frame, pos+cv::Point2f(-1, baseline),
                  pos+cv::Point2f(text_size.width+1, -text_size.height-6),col, 2);
    pos.x+=1;
    cv::putText(frame, text, pos, fontface, scale, cv::Scalar(255,255,255), thickness, 8);
    pos.x+=1;

    text = "O: next frame";
    pos.x += text_size.width+4;
    text_size = cv::getTextSize(text, fontface, scale, thickness, &baseline);
    cv::rectangle(frame, pos+cv::Point2f(-1, baseline),
                  pos+cv::Point2f(text_size.width+1, -text_size.height-6),col, 2);
    pos.x+=1;
    cv::putText(frame, text, pos, fontface, scale, cv::Scalar(255,255,255), thickness, 8);


    // write additional information if more_info is true
    if(more_info){
        // draw background
        alpha=0.8;
        cv::Mat overlay;
        frame.copyTo(overlay);
        cv::rectangle(overlay, cv::Point2f((13*24)+22, frame.rows-24),
                      cv::Point2f(frame.cols-4, frame.rows-(4*24)+4), black_bg, -1);
        cv::addWeighted(overlay, alpha, frame, 1-alpha, 0, frame);

        pos.x = 13*24 + 22;
        pos.y -= 24*3;
        int nr=0;
        for(int e=ids.size()-1; e>=0; --e){
            if(confidence[e]<90000 && nr<6) {
                if(ids[e]==-1)
                    text = "unk.";
                else
                    text = id_to_text(ids[e]);
                int sim = (int) confidence[e];
                text += ": " + std::to_string(sim);
                if (confidence[e] < min_sim) col = green_box;
                else col = red_box;
                if (nr == 3) {
                    pos.x += 150;
                    pos.y -= 24 * 3;
                }
                cv::putText(frame, text, pos, fontface, scale, col, thickness, 8);
                pos.y += 24;
                ++nr;
            }
        }
    }

    text = "Rank: " + std::to_string(rank);
    if(rank>21) col=light_red;
    else if(rank<21) col=light_green;
    else col=green_box;
    cv::putText(frame, text, cv::Point2i(0,60), fontface, scale, col, thickness, 8);
}

#endif
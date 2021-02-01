#ifndef FELIBS_H
#define FELIBS_H

#include <iostream>
#include <fstream>
#include <sstream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>


// calculate otsu threshold for given image with 'levels' range of pixel values
int otsu_threshold(cv::Mat image, int levels) {
	// Calculate histogram frequencies
    std::vector<float> nk(levels); // frequencies
    float total = image.rows * image.cols; // total number of pixels
    float m_G = 0; // global mean

    // scan every pixel and count occurrences of intensity levels
    if(image.isContinuous()){ // if image is continuous
        unsigned char* ptr = image.ptr(0);
        unsigned char* end = ptr + (image.rows * image.cols);

        while( ptr < end ){
            nk[*ptr]++;
            ptr++;
        }
    }
    else{ // if image is not continuous
    	for(int j=0; j<image.rows; ++j){
            unsigned char* ptr = image.ptr(j);
            unsigned char* end = ptr + image.cols;

            while( ptr < end ){
                nk[*ptr]++;
                ptr++;
            }
        }
    }

    // Normalize histogram and calculate global mean (m_G)
    for(int i=0; i<levels; ++i){
        nk[i] = nk[i] / total;
        m_G += i * nk[i];
    }

    // Find the otsu threshold based on between variance for growing Ts
    float P1_T = 0; // cumulative sum calculated on the run
    float m_T = 0; // cumulative mean calculated on the run
    float between = 0; // between variance
    float maximum = 0; // max between-variance found so far
    int otsu_level = 0; // level of max between-variance found so far

    // for all possible thresholds T=i
    for(int i=0; i<levels; ++i)
    {
        P1_T += nk[i]; // update cumulative sum for all Ts up to i
        m_T += i * nk[i]; // update cumulative mean for all Ts up to i

        // calculate between variance of threshold T=i
        between = ( ((m_G*P1_T) - m_T)*((m_G*P1_T) - m_T) ) / ( P1_T*(1-P1_T) );

        // if it is greater than previous max, then save actual level i
        if( between > maximum ){
            maximum = between;
            otsu_level = i;
        }
    }
    nk.clear();
    return otsu_level;
}

// improved by receiving in input a pointer in which result is placed
// transform image in a black-white image based on threshold
void apply_threshold_gray(cv::Mat* input, int threshold){
    // scan every pixel and assign 0 or 255 intensity based on threshold
    if(input->isContinuous()){ // if image is continuous
        unsigned char* ptr = input->ptr(0);
        unsigned char* end = ptr + (input->rows * input->cols);

        while( ptr < end ){
            if(*ptr < threshold)
                *ptr = 0;
            else
                *ptr = 255;
            ptr++;
        }
    }
    else{ // if image is not continuous
        for(int j=0; j<input->rows; ++j){
            unsigned char* ptr = input->ptr(j);
            unsigned char* end = ptr + input->cols;
            //unsigned char* outputPtr = output.ptr(j);

            while( ptr < end ){
                if(*ptr < threshold)
                    *ptr = 0;
                else
                    *ptr = 255;
                ptr++;
            }
        }
    }
}

// merge input_a and input_b based on threshold_image(black-white)
cv::Mat merge_on_threshold(cv::Mat input_a, cv::Mat input_b, cv::Mat thresholded_image){

    cv::Mat output(thresholded_image.size(), CV_8UC1); // same dimension

    // scan every pixel and count occurrences of intensity
    if(thresholded_image.isContinuous()){ // if image is continuous
        unsigned char* thr_ptr = thresholded_image.ptr(0);
        unsigned char* end = thr_ptr + (thresholded_image.rows * thresholded_image.cols);

        unsigned char* in_a = input_a.ptr(0);
        unsigned char* in_b = input_b.ptr(0);
        unsigned char* out = output.ptr(0);

        while( thr_ptr < end ){
            if(*thr_ptr == 0)
                *out = *in_a;
            else // *thr_ptr == 255
                *out = *in_b;
            thr_ptr++;
            in_a++;
            in_b++;
            out++;
        }
    }
    else{ // if image is not continuous
        for(int j=0; j<thresholded_image.rows; ++j){
            unsigned char* thr_ptr = thresholded_image.ptr(j);
            unsigned char* end = thr_ptr + thresholded_image.cols;

            unsigned char* in_a = input_a.ptr(j);
            unsigned char* in_b = input_b.ptr(j);
            unsigned char* out = output.ptr(j);

            while( thr_ptr < end ){
                if(*thr_ptr == 0)
                    *out = *in_a;
                else // *thr_ptr == 255
                    *out = *in_b;
                thr_ptr++;
                in_a++;
                in_b++;
                out++;
            }
        }
    }
    return output;
}

// adds 'padding' pixels (set to 0) of padding around the image
cv::Mat add_padding(cv::Mat input, int padding){
    cv::Mat output(input.rows+(2*padding), input.cols+(2*padding), input.type());
    unsigned int k = 0;
    unsigned char* in_ptr;

    // add zeros padding at border
    for(int j=0; j<output.rows; ++j){
        unsigned char* out_ptr = output.ptr(j);
        unsigned char* end = out_ptr + output.cols;
        unsigned int i = 0;
        if(j >= padding && j <= input.rows+padding){
            in_ptr = input.ptr(k);
            k++;
        }
        while( out_ptr < end ){
            if(j < padding || i < padding || j >= input.rows+padding || i >= input.cols+padding) {
                *out_ptr = 0;
            }else {
                *out_ptr = *in_ptr;
                in_ptr++;
            }
            i++;
            out_ptr++;
        }
    }
    return output;
}

// for debugging: prints neighbor of a given point. used for debugging
void print_neighbor(cv::Mat input, int r, int c, int size){
    for(int row=r-size; row<r+size+1; row++) {
        for (int col=c-size; col<c+size+1; col++) {
            if(row>=0 && col>=0 && row<input.rows && col<input.cols) {
                unsigned char *ptr = input.ptr(row);
                ptr += col;
                std::cout << (*ptr == 255) << " ";
            }
        }
        std::cout << std::endl;
    }
}

/***
 * searches for the first uppermost, leftmost white pixel not seen yet.
 * starts searching from coordinates (x, y).
 * when encountering a boundary already seen, skips white pixels until the next black one,
 * after which, continues the search.
 * @param input gray scale image
 * @param history vector of boundary points already seen
 * @param starting(x, y) coordinates at which the search should start
 * @return first uppermost, leftmost boundary point
 */
cv::Point2i find_white(cv::Mat input, std::vector<cv::Point2i> history, cv::Point2i starting){
    bool found = false; // found white pixel
    bool seen = false; // found an already seen boundary white pixel
    int row = starting.y; // starting row
    cv::Point2i pt(0, 0);
    // check all rows starting from j
    while(row < input.rows && !found){
        unsigned char* ptr = input.ptr(row);
        unsigned char* end = ptr + input.cols;
        ptr = ptr + starting.x; // moves to (x, y)
        unsigned int col = starting.x;

        while( ptr < end && !found){
            if(*ptr == 255){ // found white pixel
                pt = cv::Point2i(col, row);
                // check that it is the first time we see that pixel
                found = true;
                for(int k=0; k < history.size() && found; k++) {
                    // in case it has been already seen set 'seen' = true
                    if (history[k].x == col && history[k].y == row) {
                        found = false;
                        seen = true;
                        starting.x = 0;
                    }
                }
            }

            // in case we encountered a boundary pixel already seen, skip the following white pixels
            if(seen){
                while(ptr < end && *ptr == 255) {
                    ptr++;
                    col++;
                }
                seen = false;
            }
            else {
                ptr++;
                col++;
            }
        }
        row++;
    }
    if(found) return pt;
    else { // in case we are at the end of file return a point out of boundaries
        pt = cv::Point2i(input.cols, input.rows);
        return pt;
    }
}

// calculates boundaries using Moore algorithm
std::vector<std::vector<cv::Point2i>> moore_boundary(cv::Mat input, int min_perimeter, int max_perimeter){
    cv::Point2i c0, b0(0, 0);
    int cd, rd;
    int perimeter = 0;
    bool clocked; // true when we make a complete clockwise spin (useful for single white pixels)
    bool found; // true when we find the next boundary point
    std::vector<cv::Point2i> points_seen; // takes track of current components boundary points
    std::vector<cv::Point2i> history; // takes track of all boundary points
    std::vector<std::vector<cv::Point2i>> big_regions; // takes track only of required length components
    // stops when receive a point out of the image
    while(b0.y < input.rows && b0.x < input.cols){
        perimeter = 0;
        // find next uppermost leftmost boundary point not seen yet
        b0 = find_white(input, history, b0);
        if(b0.y >= input.rows && b0.x >= input.cols) {
            clocked = true;
            history.clear();
            return big_regions;
        }
        else {
            points_seen.push_back(cv::Point2i(b0.x, b0.y));
        }

        cv::Point2i bi = b0;
        cv::Point2i c1, old;
        // xd, yd express shifting coordinates (from bi) in which we should search.
        // they move in clockwise direction
        cd = -1; // column displacement
        rd = 0;  // row displacement
        c0.y = b0.y + rd;
        c0.x = b0.x + cd; // west neighbor of b0
        c1 = c0;
        old = c0;
        // follow boundary
        do {
            // check for a 255 in neighbours in clockwise direction
            clocked = false;
            found = false;
            while (!clocked && !found) {
                // first half of clock
                while ((rd < 0 || (rd == 0 && cd == 1)) && !clocked && !found) {
                    if (input.at<unsigned char>(bi.y+rd, bi.x+cd) == 255) {
                        bi.x = bi.x+cd;
                        bi.y = bi.y+rd;
                        points_seen.push_back(cv::Point2i(bi.x, bi.y));

                        c0 = old;
                        cd = c0.x-bi.x;
                        rd = c0.y-bi.y;
                        found = true;
                        perimeter++;
                    } else {
                        old.x = bi.x+cd;
                        old.y = bi.y+rd;
                        if (cd < 1) ++cd;
                        else ++rd;
                        c1.x = bi.x+cd;
                        c1.y = bi.y+rd;
                    }

                    if (c0 == c1) clocked = true;
                }
                // second half of clock
                while ((rd > 0 || (rd == 0 && cd == -1)) && !clocked && !found) {
                    if (input.at<unsigned char>(bi.y+rd, bi.x+cd) == 255) {
                        bi.x = bi.x+cd;
                        bi.y = bi.y+rd;
                        points_seen.push_back(cv::Point2i(bi.x, bi.y));

                        c0 = old;
                        cd = c0.x-bi.x;
                        rd = c0.y-bi.y;
                        found = true;
                        perimeter++;
                    } else {
                        old.x = bi.x+cd;
                        old.y = bi.y+rd;
                        if (cd > -1) --cd;
                        else --rd;
                        c1.x = bi.x+cd;
                        c1.y = bi.y+rd;
                    }

                    if (c0 == c1) clocked = true;
                }
            }
        } while (bi != b0 && !clocked);

        // save only those points with required length
        if (perimeter >= min_perimeter && perimeter <= max_perimeter) {
            //big_regions.push_back(points_seen.begin(), points_seen.end());
            big_regions.push_back(points_seen);
        }
        history.insert(history.end(), points_seen.begin(), points_seen.end());
        points_seen.clear();
    }
    //for(int k=0; k < big_regions.size(); k++) output.at<unsigned char>(big_regions[k].x, big_regions[k].y) = 255;
    history.clear();
    return big_regions;
}


std::vector<std::vector<cv::Point2i>> moore_boundary(cv::Mat input, int min_perimeter){
    return moore_boundary(input, min_perimeter, (input.rows+input.cols)*2);
}


// fill internal holes with white
cv::Mat fill_holes(cv::Mat input){
    cv::Mat im_floodfill = input.clone();
    floodFill(im_floodfill, cv::Point(0,0), cv::Scalar(255));

    // Invert floodfilled image
    cv::Mat im_floodfill_inv;
    bitwise_not(im_floodfill, im_floodfill_inv);

    // Combine the two images to get the foreground.
    cv::Mat output;
    output = (input | im_floodfill_inv);

    im_floodfill.release();
    im_floodfill_inv.release();

    return output;
}

// applies Hough Transform and returns (r, theta) for which H[r, theta] is max, twice
std::vector<cv::Point2d> apply_hough_transform(cv::Mat input){
    int slices = 180;
    std::vector<cv::Point2d> result;
    cv::Mat space(1400, slices, input.type(), cv::Scalar(0,0,0));
    int maxe = 0;
    cv::Point2d max_point(0, 0);
    cv::Point2d max2(0, 0);

    int y = 0;
    double slice=CV_PI/slices, r=0;
    unsigned char* ptr, *end, *spacePtr;
    for(int x=0; x<input.rows; ++x){
        y = 0;
        r = 0;
        ptr = input.ptr(x);
        end = ptr + input.cols;
        spacePtr;

        while( ptr < end ) {
            if( *ptr == 255 ) {
                for (int theta = 0; theta <= slices; ++theta) {
                    r = (x * cos(slice * theta)) + (y * sin(slice * theta));
                    spacePtr = space.ptr(cvRound(r) + 500);
                    spacePtr += theta;
                    *spacePtr += 1;
                    if( *spacePtr > maxe ) {
                        maxe = *spacePtr;
                        max_point.x = cvRound(r);
                        max_point.y = theta*slice;
                    }
                }
            }
            ptr++;
            y++;
        }
    }
    maxe = 0;
    for(int y=0; y<space.rows; ++y){
        for(int x=0; x<space.cols; ++x) {

            spacePtr = space.ptr(y, x);
            if (*spacePtr > maxe) {
                maxe = *spacePtr;
                max2.x = max_point.x;
                max2.y = max_point.y;
                max_point.x = y - 500;
                max_point.y = x * slice;
            }
        }
    }
    result.push_back(max_point);
    result.push_back(max2);

    space.release();
    return result;
}

#endif //FELIBS_H
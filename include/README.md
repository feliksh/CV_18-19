FeLibs.h contains my implementation of computer vision algorithms: Otsu Thresholding, Moore Boundary, Fill Holes, Hough Transform.  
Macros.h contains functions used for the project: 
* _extract_cards_: used to extract all 52 cards present in cards.jpg, save them in cards/, detect all 13 ranks and save them in ranks/, detect all 4 seeds and save them in seeds/.
* _detect_rank/seed_: algorithms used to detect the most similar rank/seed to a given input card. 
Constraint: do not use Deep Learning or any special ML algorithm, otherwise it's too easy
* _calculate_probabilities_: used to compute win probability for a newly drawn card

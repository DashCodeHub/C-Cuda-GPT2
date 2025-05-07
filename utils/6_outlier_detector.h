/*
Track a moving window of values (like losses or gradient norms during training)
and detect if the newest value is an "outlier" — meaning significantly different from past values.
 It computes a Z-score:

If Z-score is large → the value is "weird" compared to previous ones → possible problem 
(diverging loss, exploding gradient, etc).

It is very lightweight:No dynamic memory, No expensive recalculation at each step, Just sliding window mean and variance.
*/

#include <stdio.h>
#include <math.h>

// window size is 128 entries
#define OUTLIER_DETECTOR_WINDOW_SIZE 128

typedef struct {
    double buffer[OUTLIER_DETECTOR_WINDOW_SIZE];
    int count;
    int index;
    double sum;
    double sum_sq;
} OutlierDetector;

void init_detector(OutlierDetector *detector) {
    for (int i = 0; i < OUTLIER_DETECTOR_WINDOW_SIZE; i++) {
        detector->buffer[i] = 0.0;
    }
    detector->count = 0;
    detector->index = 0;
    detector->sum = 0.0;
    detector->sum_sq = 0.0;
}

double update_detector(OutlierDetector *detector, double new_value) {

    if (detector->count < OUTLIER_DETECTOR_WINDOW_SIZE) {
        // here we are still building up a window of observations
        detector->buffer[detector->count] = new_value;
        detector->sum += new_value;
        detector->sum_sq += new_value * new_value;
        detector->count++;
        return nan(""); // not enough data yet

    } else {
        // we've filled the window, so now we can start detecting outliers

        // pop the oldest value from the window
        double old_value = detector->buffer[detector->index];
        detector->sum -= old_value;
        detector->sum_sq -= old_value * old_value;
        // push the new value into the window
        detector->buffer[detector->index] = new_value;
        detector->sum += new_value;
        detector->sum_sq += new_value * new_value;
        // move the index to the next position
        detector->index = (detector->index + 1) % OUTLIER_DETECTOR_WINDOW_SIZE;
        // calculate the z-score of the new value
        double mean = detector->sum / OUTLIER_DETECTOR_WINDOW_SIZE;
        double variance = (detector->sum_sq / OUTLIER_DETECTOR_WINDOW_SIZE) - (mean * mean);
        double std_dev = sqrt(variance);
        if (std_dev == 0.0) {
            return 0.0;
        }
        double z = (new_value - mean) / std_dev;

        return z;
    }
}
/*
 * particle_filter.cpp
 *
 *  Based on code examples & stub from Udacity lectures
 *  created on: Dec 12, 2016 by Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
    // TODO: Set the number of particles. Initialize all particles to first position (based on estimates of
    //   x, y, theta and their uncertainties from GPS) and all weights to 1.
    // Add random Gaussian noise to each particle.
    // NOTE: Consult particle_filter.h for more information about this method (and others in this file).

    // set the number of particles
    num_particles = 128;

    // pre-allocate memory
    weights.resize(num_particles, 1.0);
    particles.resize(num_particles);
    // uniformly-distributed integer random number generator
    random_device randDev;
    // random number generator engine
    default_random_engine randGen(randDev());

    // sow initial seed
    randGen.seed(42);

    // aliases for readability
    const double std_x      = std[0];
    const double std_y      = std[1];
    const double std_theta  = std[2];

    // create normal (Gaussian) distributions for x, y & theta
    // (thus adding random Gaussian noise to each particle's x, y & theta)
    normal_distribution<double> x_normalDistr(x, std_x);
    normal_distribution<double> y_normalDistr(y, std_y);
    normal_distribution<double> theta_normalDistr(theta, std_theta);

    for (int i = 0; i < num_particles; i++) {
        Particle p;
        p.id = i;

        p.x = x_normalDistr(randGen);
        p.y = y_normalDistr(randGen);
        p.theta = theta_normalDistr(randGen);

        p.weight = 1.0 / num_particles;

        particles[i] = p;
        weights[i] = p.weight;
    }

    is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
    // TODO: Add measurements to each particle and add random Gaussian noise.
    // NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
    //  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
    //  http://www.cplusplus.com/reference/random/default_random_engine/

    // avoid repeating numeric expressions, taking computation out of loop
    const double yaw_dt     = yaw_rate * delta_t;
    const double v_dt       = velocity * delta_t;
    const double v_by_yaw   = velocity / yaw_rate;

    random_device randDev;
    // random number generator
    default_random_engine randGen(randDev());
    // sow initial seed
    randGen.seed(42);

    // aliases for readability
    const double std_pos_x      = std_pos[0];
    const double std_pos_y      = std_pos[1];
    const double std_pos_theta  = std_pos[2];

    const double epsilon = 1e-4;

    for (auto &p: particles) {

        if (fabs(yaw_rate) > epsilon) {
            const double theta_new = p.theta + yaw_dt;
            p.x += v_by_yaw * (sin(theta_new) - sin(p.theta));
            p.y += v_by_yaw * (cos(p.theta) - cos(theta_new));
            p.theta = theta_new;
        } else {
            p.x += v_dt * cos(p.theta);
            p.y += v_dt * sin(p.theta);
        }

        // create normal distributions for x, y & theta
        normal_distribution<double> x_normalDistr(p.x,          std_pos_x);
        normal_distribution<double> y_normalDistr(p.y,          std_pos_y);
        normal_distribution<double> theta_normalDistr(p.theta,  std_pos_theta);

        p.x     = x_normalDistr(randGen);
        p.y     = y_normalDistr(randGen);
        p.theta = theta_normalDistr(randGen);
    }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs> &observations) {
    // TODO: Find the predicted measurement that is closest to each observed measurement and assign the
    //   observed measurement to this particular landmark.
    // NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
    //   implement this method and use it as a helper during the updateWeights phase.

    // to compute min: init some very large number to designate upper limit
    const double UtterlyHugeNumber = 1e16;

    for (auto &iter_Observed: observations) {
        double shortest_dist = UtterlyHugeNumber;

        // the predicted measurement that is closest to each observed measurement
        LandmarkObs closestLandmark;

        for (auto iter_Predicted: predicted) {
            // compute distance between observed & predicted measurements
            double cur_dist = dist(iter_Observed.x,  iter_Observed.y,
                                   iter_Predicted.x, iter_Predicted.y);
            // if new distance is minimal then remember it in shortest_dist
            if (shortest_dist > cur_dist) {
                shortest_dist = cur_dist;
                closestLandmark = iter_Predicted;
            }
        }

        // store the new found nearest landmark
        iter_Observed = closestLandmark;
    }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   std::vector<LandmarkObs> observations, Map map_landmarks) {
    // TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
    //   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
    // NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
    //   according to the MAP'S coordinate system. You will need to transform between the two systems.
    //   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
    //   The following is a good resource for the theory:
    //   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
    //   and the following is a good resource for the actual equation to implement (look at equation
    //   3.33. Note that you'll need to switch the minus sign in that equation to a plus to account
    //   for the fact that the map's y-axis actually points downwards.)
    //   http://planning.cs.uiuc.edu/node99.html

    // aliases for readability
    const double std_x = std_landmark[0];
    const double std_y = std_landmark[1];

    // avoid repeating numeric expressions, taking computation out of loop
    const double two_std_x_sqrd = 2. * std_x * std_x;
    const double two_std_y_sqrd = 2. * std_y * std_y;
    const double two_Pi_std_x_std_y = 2. * M_PI * std_x * std_y;

    for (int i = 0; i < particles.size(); ++i) {

        Particle p = particles[i];

        vector<LandmarkObs> transformed_observations_list;

        // transform each observation coordinates from vehicle XY to map XY system
        for (auto iterObs: observations) {
            LandmarkObs transformed_observation;

            transformed_observation.id = iterObs.id;

            transformed_observation.x =
                    p.x + iterObs.x * cos(p.theta) - iterObs.y * sin(p.theta);

            transformed_observation.y =
                    p.y + iterObs.x * sin(p.theta) + iterObs.y * cos(p.theta);

            transformed_observations_list.push_back(transformed_observation);
        }

        // search for map landmarks that are nearest to the particle
        vector<LandmarkObs> predicted_landmarks;
        for (auto iterLandmark: map_landmarks.landmark_list) {

            double cur_dist = dist(p.x, p.y, iterLandmark.x_f, iterLandmark.y_f);
            if (cur_dist < sensor_range) {
                LandmarkObs nearest_landmark;

                nearest_landmark.id = iterLandmark.id_i;

                nearest_landmark.x = iterLandmark.x_f;
                nearest_landmark.y = iterLandmark.y_f;

                predicted_landmarks.push_back(nearest_landmark);
            }
        }

        // search for closest landmarks for the transformed observations
        vector<LandmarkObs> closest_landmarks;
        std::copy(transformed_observations_list.begin(), transformed_observations_list.end(),
                  std::back_inserter(closest_landmarks));
        dataAssociation(predicted_landmarks, closest_landmarks);

        // compute weights for the closest landmarks
        double wt = 1.0;
        for (int j = 0; j < closest_landmarks.size(); ++j) {
            double dx = transformed_observations_list.at(j).x - closest_landmarks.at(j).x;
            double dy = transformed_observations_list.at(j).y - closest_landmarks.at(j).y;

            wt *= 1.0 / (two_Pi_std_x_std_y) * exp(-dx * dx / (two_std_x_sqrd)) * exp(-dy * dy / (two_std_y_sqrd));

            // update stored values of the particle & filter weigths
            weights[i] = p.weight = wt;
        }
    }
}

void ParticleFilter::resample() {
    // TODO: Resample particles with replacement with probability proportional to their weight.
    // NOTE: You may find std::discrete_distribution helpful here.
    //   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

    vector<Particle> particles_resampled(num_particles);
    discrete_distribution<int> particles_discr_distrib(weights.begin(), weights.end());

    random_device randDev;
    // random number generator
    default_random_engine randGen(randDev());
    // sow initial seed
    randGen.seed(42);

    for (int i = 0; i < num_particles; ++i) {
        // take new index j at random from our distribution
        int j = particles_discr_distrib(randGen);
        // resample i-th particle
        particles_resampled[i] = particles[j];
    }
    // replace particles with their resampled peers
    particles = particles_resampled;
}

void ParticleFilter::write(std::string filename) {
    // You don't need to modify this file.
    std::ofstream dataFile;
    dataFile.open(filename, std::ios::app);
    for (int i = 0; i < num_particles; ++i) {
        dataFile << particles[i].x << " " << particles[i].y << " " << particles[i].theta << "\n";
    }
    dataFile.close();
}

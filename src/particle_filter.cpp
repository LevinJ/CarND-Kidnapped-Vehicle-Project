/*
 * particle_filter.cpp
 *
 *  Based on code examples & stub from Udacity lectures
 *  created on: Dec 12, 2016 by Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

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
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
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

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}

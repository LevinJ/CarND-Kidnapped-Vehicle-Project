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
	num_particles = 80;
	random_device rd;
	default_random_engine gen(rd());
	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);

	for(int i=0;i < num_particles; i++){
		Particle particle;
		particle.id = i;
		particle.x = dist_x(gen);
		particle.y = dist_y(gen);
		particle.theta = dist_theta(gen);
		particle.weight = 1.0;
		particles.push_back(particle);
	}
	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {

	random_device rd;
	default_random_engine gen(rd());

	normal_distribution<double> dist_x(0, std_pos[0]); //Gaussian noise initializing based on estimates by sampling from Gaussian dist
	normal_distribution<double> dist_y(0, std_pos[1]); //x,y,theta are values from GPS
	normal_distribution<double> dist_theta(0, std_pos[2]);

	for (auto &p: particles){

		if(fabs(yaw_rate) > 0.0001){
			p.x += velocity/yaw_rate* (sin(p.theta + yaw_rate * delta_t) - sin(p.theta)) + dist_x(gen);
			p.y += velocity/yaw_rate * (cos(p.theta) - cos(p.theta + yaw_rate* delta_t)) + dist_y(gen);
		}
		else{
			p.x += velocity * delta_t *cos(p.theta) + dist_x(gen);
			p.y += velocity * delta_t * sin(p.theta)+ dist_y(gen);
		}
		p.theta = p.theta + yaw_rate*delta_t + dist_theta(gen);
	}
}

std::vector<std::vector<double >>  ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs> &observations) {


}

void dataAssociationPerParticle(std::vector<LandmarkObs>& predicted, const std::vector<LandmarkObs> &observations) {

    std::vector<LandmarkObs> predicted_mapped;

    for (auto &iter_Observed: observations) {
        double shortest_dist = INFINITY;

        // the predicted measurement that is closest to each observed measurement
        LandmarkObs closestLandmark;

        for (auto iter_Predicted: predicted) {
            // compute distance between observed & predicted measurements
            double cur_dist = dist(iter_Observed.x,  iter_Observed.y,
                                   iter_Predicted.x, iter_Predicted.y);
            // if new distance is minimal then remember it in shortest_dist
            if (cur_dist < shortest_dist) {
                shortest_dist = cur_dist;
                closestLandmark = iter_Predicted;
            }
        }

        // store the new found nearest landmark
        predicted_mapped.push_back(closestLandmark);
    }
    predicted = predicted_mapped;
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	 	 weights.resize(num_particles, 1.0);

	    const double std_x = std_landmark[0];
	    const double std_y = std_landmark[1];

	    // avoid repeating numeric expressions, taking computation out of loop
	    const double two_std_x_sqrd = 2. * std_x * std_x;
	    const double two_std_y_sqrd = 2. * std_y * std_y;
	    const double two_Pi_std_x_std_y = 2. * M_PI * std_x * std_y;


	    for (int i = 0; i < particles.size(); ++i) {

	        Particle &p = particles[i];

			vector<LandmarkObs> transformed_observations;
	        // transform each observation coordinates from vehicle XY to map XY system
	        for (int j=0; j< observations.size(); j++) {
	        	double x =p.x + observations[j].x * cos(p.theta) - observations[j].y * sin(p.theta);
	        	double y =p.y + observations[j].x * sin(p.theta) + observations[j].y * cos(p.theta);
				transformed_observations.push_back(LandmarkObs{observations[j].id, x, y});
	        }

	        vector<LandmarkObs> predicted_landmarks;
	        for (auto iterLandmark: map_landmarks.landmark_list) {

				LandmarkObs landmark;
				landmark.id = iterLandmark.id_i;
				landmark.x = iterLandmark.x_f;
				landmark.y = iterLandmark.y_f;
				predicted_landmarks.push_back(landmark);
	        }


	        dataAssociationPerParticle(predicted_landmarks, transformed_observations);
	        // compute weights for the closest landmarks
	        std::vector<int> associations;
			std::vector<double> sense_x;
			std::vector<double> sense_y;
	        double wt = 1.0;
	        for (int j = 0; j < predicted_landmarks.size(); ++j) {
	        	associations.push_back(predicted_landmarks[j].id);
	        	sense_x.push_back(transformed_observations[j].x);
	        	sense_y.push_back(transformed_observations[j].y);
	            double dx = transformed_observations[j].x - predicted_landmarks[j].x;
	            double dy = transformed_observations[j].y - predicted_landmarks[j].y;

	            wt *= 1.0 / (two_Pi_std_x_std_y) * exp(-dx * dx / (two_std_x_sqrd)) * exp(-dy * dy / (two_std_y_sqrd));

	            // update stored values of the particle & filter weigths
	            weights[i] = p.weight = wt;
	        }

	        p = SetAssociations(p, associations, sense_x, sense_y);
	    }
}

void ParticleFilter::resample() {
    vector<Particle> resample_particles;
    discrete_distribution<int> distribution(weights.begin(), weights.end());
    random_device rd;
    default_random_engine randGen(rd());
    for (int i = 0; i < num_particles; ++i) {
    	resample_particles.push_back(particles[distribution(randGen)]);
    }
    particles = std::move(resample_particles);
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations,
        const std::vector<double>& sense_x, const std::vector<double>& sense_y)
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

/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// Set the number of particles. Initialize all particles to first position (based on estimates of
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

	num_particles = 1000;

	double std_x = std[0];
	double std_y = std[1];
	double std_theta = std[2];

	random_device rd;
	default_random_engine gen(rd());
	normal_distribution<double> dist_x(x, std_x);
	normal_distribution<double> dist_y(y, std_y);
	normal_distribution<double> dist_theta(theta, std_theta);

	for(int i=0;i < num_particles; i++){
		Particle particle;
		particle.id = i;
		particle.x = dist_x(gen);
		particle.y = dist_y(gen);
		particle.theta = dist_theta(gen);
		particle.weight = 1.0;
		particles.push_back(particle);
		weights.push_back(particle.weight);
		cout << "Sample " << particle.id  << " " << particle.x << " " << particle.y << " " << particle.theta<< endl;

	}



	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	for(int i=0; i< num_particles;i++){
		Particle &particle = particles[i];
		double new_theta = particle.theta + yaw_rate * delta_t;
		particle.x = particle.x + (velocity/yaw_rate) * (sin(new_theta) - sin(particle.theta));
		particle.y = particle.y + (velocity/yaw_rate) *(cos(particle.theta) -  cos(new_theta));
		particle.theta = new_theta;
		//maybe need to add noise(std_pos) later on, though for now I don't need to do so
	}

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
	//	for(int i=0; i< num_particles;i++){
	//		Particle &particle = particles[i];
	//		for (int j =0; j< observations.size();j++){
	//			LandmarkObs& landmarkobs = observations[j];
	//
	//
	//		}
	//
	//
	//	}


}

void dataAssociationPerParticle(std::vector<LandmarkObs>& predicteds, const std::vector<LandmarkObs>& observations,
		const Map& map_landmarks, const Particle& particle){
	//find predicted landmark measurements corresponding to a specific particle and actual landmark measurments
	for (int i =0; i< observations.size();i++){
		const LandmarkObs& landmarkobs = observations[i];

		//transform landmark observation to map's coordinate system
		LandmarkObs landmarkobs_transformed;
		landmarkobs_transformed.x = particle.x + landmarkobs_transformed.x;
		landmarkobs_transformed.y = particle.y + landmarkobs_transformed.y;
		//Find closet landmark as the predicted landmark
		double clostest_dist = -1;
		int predicted_landmark_id = -1;
		for(int j=0; j< map_landmarks.landmark_list.size(); j++){
			const Map::single_landmark_s& landmark_candidate = map_landmarks.landmark_list[j];
			double x_dist = landmark_candidate.x_f - landmarkobs_transformed.x;
			double y_dist = landmark_candidate.y_f - landmarkobs_transformed.y;
			double dist = sqrt(x_dist*x_dist + y_dist*y_dist);

			if(clostest_dist == -1 || dist < clostest_dist ){
				clostest_dist = dist;
				predicted_landmark_id = j;
			}
		}
		//transform predicted landmark to vehicle coordinate system
		const Map::single_landmark_s& landmark_closet = map_landmarks.landmark_list[predicted_landmark_id];
		LandmarkObs predicted_measurement;
		predicted_measurement.id = landmark_closet.id_i;
		predicted_measurement.x = landmark_closet.x_f - particle.x;
		predicted_measurement.y = landmark_closet.y_f - particle.y;
		predicteds.push_back(predicted_measurement);
	}

}

double comupte_bivariate_gaussian(const std::vector<LandmarkObs>& observations, const std::vector<LandmarkObs>& predicteds,double std_landmark[]){
	//compute bivariate-gaussian using the link https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	//under section Bivariate case
	double weight_product = 0;
	double sigma_x = std_landmark[0];
	double sigma_y = std_landmark[1];
//	double p = 0;

	for(int i=0; i< observations.size();i++){
		//Compute predicted landmark measurement for the particle
		const LandmarkObs& observation = observations[i];
		const LandmarkObs& predicted = predicteds[i];
		double x = observation.x;
		double y = observation.y;

		double mu_x = predicted.x;
		double mu_y = predicted.y;

		double gap_x = x - mu_x;
		double gap_y = y - mu_y;

		double weight_1 = 1.0/(2 * M_PI * sigma_x * sigma_y);
		double weight_2 = pow(gap_x, 2)/pow(sigma_x, 2) + pow(gap_y,2)/pow(sigma_y, 2);
		weight_2 = exp(-0.5 * weight_2);
		double weight = weight_1 * weight_2;

		//multiply density for all predicted measurements
		if(weight_product == 0){
			weight_product = weight;
		}else{
			weight_product = weight_product *weight;
		}
	}
	return weight_product;

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



	for(int i=0; i< num_particles;i++){
		//Compute predicted landmark measurement for the particle
		Particle &particle = particles[i];
		std::vector<LandmarkObs> predicteds;
		dataAssociationPerParticle(predicteds, observations,map_landmarks, particle);
		//TODo, may need to add sensor_range handling later on
		particle.weight = comupte_bivariate_gaussian(observations, predicteds,std_landmark);
		cout<<"weigth for particle" << particle.id << ", " << particle.weight << endl;
	}






}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

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

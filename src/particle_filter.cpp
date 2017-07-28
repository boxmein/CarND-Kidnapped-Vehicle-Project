/*
 * particle_filter.cpp
 *
 *  Created on: Jul 28, 2017
 *      Author: Johannes Kadak (based on skeleton by Tiffany Huang)
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

#include <climits>

#include "particle_filter.h"

using namespace std;

const double epsilon = 1e-4;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of
	//   x, y, theta and their uncertainties from GPS) and all weights to 1.
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	
	std::default_random_engine rng;
	
	std::normal_distribution<double> x_dist(x, std[0]);
	std::normal_distribution<double> y_dist(y, std[1]);
	std::normal_distribution<double> theta_dist(theta, std[2]);
	
	num_particles = 1000;
	
	for (int i = 0; i < num_particles; i++) {
		Particle p;
		p.id = i;
		p.x = x_dist(rng);
		p.y = y_dist(rng);
		p.theta = theta_dist(rng);
		p.weight = 1;
		
		particles.push_back(p);
	}
	
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	
	std::default_random_engine rng;
	
	std::normal_distribution<double> x_dist(0, std_pos[0]);
	std::normal_distribution<double> y_dist(0, std_pos[1]);
	std::normal_distribution<double> theta_dist(0, std_pos[2]);
	
	std::cout << "v: " << velocity << " yawr: " << yaw_rate << std::endl;
	
	for (Particle& p : particles) {
		
		if (fabs(yaw_rate) < epsilon) {
			// yaw rate is practically 0
			std::cout << "!!! yaw rate practically 0 !!!" << std::endl;
		} else {
			const double v_yr = velocity / yaw_rate;
			const double theta2 = p.theta + yaw_rate * delta_t;
			
			// Use the bicycle motion model equations + add noise
			p.x += v_yr * ( sin(theta2) - sin(p.theta) ) + x_dist(rng);
			p.y += v_yr * ( cos(p.theta) - cos(theta2) ) + y_dist(rng);
			
			p.theta = theta2 + theta_dist(rng);
		}
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
	//   implement this method and use it as a helper during the updateWeights phase.
	
	// predicted - landmark's relative coordinates for particle
	// observed  - lidar saw a landmark
	
	for (LandmarkObs &ob : observations) {
		
		LandmarkObs min_landmark;
		double min_dist = 1e12;
		
		// Find the closest predicted landmark
		for (LandmarkObs &pr : predicted) {
			double dx = pr.x - ob.x;
			double dy = pr.y - ob.y;
			double dist = sqrt(dx * dx + dy * dy);
				
			if (dist < min_dist) {
				min_landmark = pr;
				min_dist = dist;
			}
		}
		
		// Use the closest predicted landmark's ID as the observation's ID
		ob.id = min_landmark.id;
	}
}


// Transform a landmark observation from local to global coordinates,
// depending on particle's global coordinates.
LandmarkObs transformLandmark(LandmarkObs observation, Particle p) {
	LandmarkObs ret;
	
	ret.id = observation.id;
	
	// Rotation
	ret.x = observation.x * cos(p.theta) - observation.y * sin(p.theta);
	ret.y = observation.x * sin(p.theta) + observation.y * cos(p.theta);

	// Transformation
	ret.x += p.x;
	ret.y += p.y;

	return ret;
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
	
	
	for (Particle &p : particles) {
		p.weight = 1;
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight.
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	std::default_random_engine rng;
	std::discrete_distribution<int> particleDist(weights.begin(), weights.end());
	
	std::vector<Particle> particles_new;

	for (int i = 0; i < 1000; i++) {
		particles_new.push_back( particles[ particleDist(rng) ] );
	}
	
	particles.clear();
	particles = particles_new;
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

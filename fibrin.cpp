#include <iostream>
#include <complex>
#include <fstream>
#include <string>
#include <math.h>
#include <random>
#include <iomanip>
#include <cmath>
#include <map>
#include <sys/stat.h>
#include <vector>
#include <numeric>
#include <gsl/gsl_math.h>
#include <gsl/gsl_multimin.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_odeiv.h>
#include <gsl/gsl_rng.h>

using namespace std;

// The model parameters

// The following values determine the units:
const double R = 1; // spherocylinder radius
const double E_0 = 1 * R * R; // elastic modulus of the cell
const double A = 1;	// growth rate

// Cell-cell and cell-surface interaction parameters:
//const double A_0 = E_0/1000.0; 	// cell-cell adhesion energy
const double A_0 = 0;
double E_1 = 100;  	// the elastic modulus of cell-surface interaction (cylindrical term)
const double A_1 = E_1/1000.0; 	// cell-surface adhesion energy (cylindrical term)
const double E_2 = E_1;			// the elastic modulus of cell-surface interaction (spherical term)
const double A_2 = A_1;			// cell-surface adhesion energy (spherical term)

// Viscosity parameters:
double NU_1 = 0.1 * (E_0) / 1; // surface drag coefficient
double NU_0 = NU_1 * E_0 / 100; // ambient (positional) drag

// Additional cell parameters:
const double Adev  = 0.2; // growth rate noise
const double L0 = 2*R; // initial length of the first cell
double Lf; // mean final length

// Simulation parameters
const double DOF_GROW = 7; // degrees of freedom (growing cell)
const int MAX_DOF = 5000; // Total allowed degrees of freedom. N * DOF_GROW must be less than MAX_DOF, else segfault!
const double noise_level = 1e-8; // A small, symmetry breaking noise is chosen from a distribution of this width
// and added to the generalized forces at each step during the evolution.

const double h0 = 1e-4; // absolute accuracy of diff. eq. solver

const double T = 12; // Integrate for this amount of total time.
const double tstep = 0.1; // for output purposes only. Outputs data after every interval of time "tstep"

// Note: pi is given by M_PI

double xy = 0; // Confine to xy plane?
const double xz = 0; // Confine to xz plane?

const int D = 5; // The physical degrees of freedom of a single cell.

const double hd0 = 0.01;
const double k0 = 4.5;

// Random number generation
std::default_random_engine generator;
std::normal_distribution<double> normal(0, Adev);

// File name for output
string my_name;

struct overlapVec {
	// For two contacting cells, this struct specifies the locations of the points along
	// the cell centerlines that are separated by the smallest distance.
	// The distance is given by "overlap."
    double rix1, riy1, riz1;
	double rix2, riy2, riz2;
	double overlap;
};
struct myVec {
	// Vector
	double x, y, z;
};
struct my6Vec {
	// Cell center-of-mass position (x, y, z) and orientation (nx, ny, nz)
	double x, y, z, nx, ny, nz;
};
struct derivFunc{
	// This struct is used when numerically integrating the equation of motion
	// to store the generalized forces acting on the cell coordinates
	double xd, yd, zd, nxd, nyd, nzd;
};
struct pop_info {
	// When tracking the verticalization of a cell,
	// this struct is used to store the orientation and
	// cell-to-cell contact forces provided by neighboring cells
	double vx, vy, vz;
	double fx, fy, fz;
	double nz;
};

double cot(double i) { return(1 / tan(i)); } // Cotangent function
double csc(double i) { return(1 / sin(i)); } // Cosecant function

double cross(double x1, double y1, double z1, double x2, double y2, double z2, int dof) {
	// Cross product of r1 = (x1, y1, z1) and r2 (x2, y2, z2).
	// Returns the degree of freedom "dof" of the vector r1 x r2
	if (dof == 0) {
		return -(y2*z1) + y1*z2;
	} else if (dof == 1) {
		return x2*z1 - x1*z2;
	} else if (dof == 2) {
		return -(x2*y1) + x1*y2;
	}
	return 0;
}

double vnorm(double x, double y, double z) {
	// Return the magnitude of a vector (x, y, z)
	return sqrt(x*x + y*y + z*z);
}

class Cell {
	// One cell. Stores the coordinates and creates daughter cells when time to divide
	private:
		double x, y, z, nx, ny, nz, l, Lfi, ai; //initial node position on grid
		double tborn;
		string lineage;
		int fpop;
		double current_nz;
		int current_sa;
		
		double dx, dy, dz, dnx, dny, dnz;
		double m1x, m1y, m1z, m2x, m2y, m2z;
		
		double mf1, mf2, mf3, max_l;
		double of1, of2, of3;
		int confine;
		//Cell* d1_cell;
		//Cell* d2_cell;
		
		vector<pop_info> pop_vec;
		
	public:
		Cell (double, double, double, double, double, double, double);
		void set_pos(double mx, double my, double mz) {
			x = mx;
			y = my;
			z = mz;
		}
		void set_n(double mnx, double mny, double mnz) {
			nx = mnx;
			ny = mny;
			nz = mnz;
		}
		void set_l(double ml) {
			l = ml;
		}
		void set_confine() { confine = 1; }
		void set_ai(double mai) { ai = mai; }
		void set_x(double mx) { x = mx; }
		void set_y(double mx) { y = mx; }
		void set_z(double mx) { z = mx; }
		void set_nx(double mx) { nx = mx; }
		void set_ny(double mx) { ny = mx; }
		void set_nz(double mx) { nz = mx; }
		void set_m() {
			double ax = rand()/(static_cast<double>(RAND_MAX));
			double ay = rand()/(static_cast<double>(RAND_MAX));
			double az = rand()/(static_cast<double>(RAND_MAX));
			double cx = cross(nx, ny, nz, ax, ay, az, 0);
			double cy = cross(nx, ny, nz, ax, ay, az, 1);
			double cz = cross(nx, ny, nz, ax, ay, az, 2);
			double cn = vnorm(cx, cy, cz);
			while (cn < 1e-6) {
				ax = rand()/(static_cast<double>(RAND_MAX));
				ay = rand()/(static_cast<double>(RAND_MAX));
				az = rand()/(static_cast<double>(RAND_MAX));
				cx = cross(nx, ny, nz, ax, ay, az, 0);
				cy = cross(nx, ny, nz, ax, ay, az, 1);
				cz = cross(nx, ny, nz, ax, ay, az, 2);
				cn = vnorm(cx, cy, cz);
			}
			m1x = cx / cn;
			m1y = cy / cn;
			m1z = cz / cn;
			
			m2x = cross(nx, ny, nz, m1x, m1y, m1z, 0);
			m2y = cross(nx, ny, nz, m1x, m1y, m1z, 1);
			m2z = cross(nx, ny, nz, m1x, m1y, m1z, 2);
			
			/*
			m1x = 0;
			m1y = 1;
			m1z = 0;
			
			m2x = 0;
			m2y = 0;
			m2z = 1;
			*/
		}
		void set_m1(double pp) {
			nx = nx + m1x * pp;
			ny = ny + m1y * pp;
			nz = nz + m1z * pp;
		}
		void set_m2(double pp) {
			nx = nx + m2x * pp;
			ny = ny + m2y * pp;
			nz = nz + m2z * pp;
		}
		void set_noise() {
		    dx = noise_level * (rand()/(static_cast<double>(RAND_MAX)) - 0.5);
		    dy = noise_level * (rand()/(static_cast<double>(RAND_MAX)) - 0.5);
		    dz = noise_level * (rand()/(static_cast<double>(RAND_MAX)) - 0.5);
		    dnx = noise_level * (rand()/(static_cast<double>(RAND_MAX)) - 0.5);
		    dny = noise_level * (rand()/(static_cast<double>(RAND_MAX)) - 0.5);
		    dnz = noise_level * (rand()/(static_cast<double>(RAND_MAX)) - 0.5);
		}
		void set_tborn(double ti) {
			tborn = ti;
		}
		void set_lin(string xb) {
			lineage = lineage + xb;
		}
		void set_fpop() { fpop = 1; }
		void set_current_nz(double mz) {
			current_nz = mz;
		}
		void set_current_sa(int sa) { current_sa = sa; }
		void set_mf1(double mf) {
			mf1 = mf;
		}
		void set_mf2(double mf) {
			mf2 = mf;
		}
		void set_mf3(double mf) {
			mf3 = mf;
		}
		void set_of1(double mf) {
			of1 = mf;
		}
		void set_of2(double mf) {
			of2 = mf;
		}
		void set_of3(double mf) {
			of3 = mf;
		}
		void set_max_l(double mf) {
			max_l = mf;
		}
		void add_pop_vec(pop_info pi) {
			pop_vec.push_back(pi);
		}
		void clear_pop_vec() {
			pop_vec.clear();
		}
		int get_pop_contact() {
			return pop_vec.size();
		}
		double get_pop_vx(int i) {
			return pop_vec[i].vx;
		}
		double get_pop_vy(int i) {
			return pop_vec[i].vy;
		}
		double get_pop_vz(int i) {
			return pop_vec[i].vz;
		}
		double get_pop_fx(int i) {
			return pop_vec[i].fx;
		}
		double get_pop_fy(int i) {
			return pop_vec[i].fy;
		}
		double get_pop_fz(int i) {
			return pop_vec[i].fz;
		}
		double get_pop_nz(int i) {
			return pop_vec[i].nz;
		}
		double get_mf1() {
			return mf1;
		}
		double get_mf2() {
			return mf2;
		}
		double get_mf3() {
			return mf3;
		}
		double get_of1() {
			return of1;
		}
		double get_of2() {
			return of2;
		}
		double get_of3() {
			return of3;
		}
		double get_max_l() {
			return max_l;
		}
		my6Vec get_noise() {
			my6Vec noise;
			noise.x = dx;
			noise.y = dy;
			noise.z = dz;
			noise.nx = dnx;
			noise.ny = dny;
			noise.nz = dnz;
			return noise;
		}
		// Return the state of the cell
		double get_x() { return x; }
		double get_y() { return y; }
		double get_z() { return z; }
		double get_nx() { return nx; }
		double get_ny() { return ny; }
		double get_nz() { return nz; }
		double get_m1x() { return m1x; }
		double get_m1y() { return m1y; }
		double get_m1z() { return m1z; }
		double get_m2x() { return m2x; }
		double get_m2y() { return m2y; }
		double get_m2z() { return m2z; }
		double get_l() { return l; }
		double get_Lf() { return Lfi; }
		double get_ai() { return ai; }
		double get_tborn() { return tborn; }
		string get_lin() { return lineage; }
		int get_fpop() { return fpop; }
		double get_current_nz() { return current_nz; }
		int get_current_sa() { return current_sa; }
		double get_sf_axis(int dof) {
			double t0, t1, t2, a0, a1, a2, an;
			
			a0 = cross(0, 0, 1, nx, ny, nz, 0);
			a1 = cross(0, 0, 1, nx, ny, nz, 1);
			a2 = cross(0, 0, 1, nx, ny, nz, 2);
			
			an = sqrt(a0*a0 + a1*a1 + a2*a2);
			
			if (dof == 0) {
				return a0/an;
			}
			else if (dof == 1) {
				return a1/an;
			}
			else if (dof == 2) {
				return a2/an;
			} else {
				cout << "SF AXIS ERROR" << endl;
				return 0;
			}
		}
		int get_confine() { return confine; }
		
};

Cell::Cell(double mx, double my, double mz, double mnx, double mny, double mnz, double ml) {
	x = mx;
	y = my;
	z = mz;
	nx = mnx;
	ny = mny;
	nz = mnz;
	
    dx = noise_level * (rand()/(static_cast<double>(RAND_MAX)) - 0.5);
    dy = noise_level * (rand()/(static_cast<double>(RAND_MAX)) - 0.5);
    dz = noise_level * (rand()/(static_cast<double>(RAND_MAX)) - 0.5);
    dnx = noise_level * (rand()/(static_cast<double>(RAND_MAX)) - 0.5);
    dny = noise_level * (rand()/(static_cast<double>(RAND_MAX)) - 0.5);
    dnz = noise_level * (rand()/(static_cast<double>(RAND_MAX)) - 0.5);
	
	current_nz = nz;
	
	l = ml;
	
	Lfi = Lf;
		
	confine = 0;
	fpop = 0;
	
	lineage = "";
	
	ai = A + normal(generator);
}

overlapVec get_overlap_Struct(Cell* cell_1, Cell* cell_2) {
	// Returns the vector of the shortest distance from cell 1 to cell 2
	// An extension of: http://homepage.univie.ac.at/franz.vesely/notes/hard_sticks/hst/hst.html
	// dof is the component of the vector (0,1,2 for cell 1, 3,4,5 for cell 2)
	double overlap;
	
	overlapVec answer;
	
	double x1, y1, z1, nx1, ny1, nz1, l1;
	double x2, y2, z2, nx2, ny2, nz2, l2;
	double x12, y12, z12, m12, u1, u2, u12, cc, xla, xmu;
	
	double rpx, rpy, rpz, rix1, riy1, riz1, rix2, riy2, riz2;
	
	double h1, h2, gam, gam1, gam2, gamm, gamms, del, del1, del2, delm, delms, aa, a1, a2, risq, f1, f2;
	double gc1, dc1, gc2, dc2;
	
	x1 = cell_1->get_x();
	y1 = cell_1->get_y();
	z1 = cell_1->get_z();
	nx1 = cell_1->get_nx();
	ny1 = cell_1->get_ny();
	nz1 = cell_1->get_nz();
	l1 = cell_1->get_l() / 2.0;
	
	x2 = cell_2->get_x();
	y2 = cell_2->get_y();
	z2 = cell_2->get_z();
	nx2 = cell_2->get_nx();
	ny2 = cell_2->get_ny();
	nz2 = cell_2->get_nz();
	l2 = cell_2->get_l() / 2.0;
	
	x12 = x2 - x1;
	y12 = y2 - y1;
	z12 = z2 - z1;
	
	m12 = x12*x12 + y12*y12 + z12*z12;
	
	u1 = x12*nx1 + y12*ny1 + z12*nz1;
	u2 = x12*nx2 + y12*ny2 + z12*nz2;
	u12 = nx1*nx2 + ny1*ny2 + nz1*nz2;
	cc = 1.0 - u12*u12;
	
	// Check if parallel
	if (cc < 1e-6) {
	 	if(u1 && u2) {
			xla = u1/2;
			xmu = -u2/2;
		}
		else {
			// lines are parallel
			
			answer.rix1 = x1;
			answer.riy1 = y1;
			answer.riz1 = z1;
			answer.rix2 = x2;
			answer.riy2 = y2;
			answer.riz2 = z2;
			
			return answer;
	 	}
	}
	else {
		xla = (u1 - u12*u2) / cc;
	 	xmu = (-u2 + u12*u1) / cc;
	}
	
	//Rectangle half lengths h1=L1/2, h2=L2/2	
	h1 = l1; 
	h2 = l2;

	//If the origin is contained in the rectangle, 
	//life is easy: the origin is the minimum, and 
	//the in-plane distance is zero!
	if ((xla*xla <= h1*h1) && (xmu*xmu <= h2*h2)) {
	  	answer.rix1 = x1 + (xla) * nx1;
	  	answer.riy1 = y1 + (xla) * ny1;
	  	answer.riz1 = z1 + (xla) * nz1;
	  	answer.rix2 = x2 + (xmu) * nx2;
	  	answer.riy2 = y2 + (xmu) * ny2;
	  	answer.riz2 = z2 + (xmu) * nz2;
		
		return answer;
	}
	else {
	//Find minimum of f=gamma^2+delta^2-2*gamma*delta*(e1*e2)
	//where gamma, delta are the line parameters reckoned from the intersection
	//(=lam0,mu0)

	//First, find the lines gamm and delm that are nearest to the origin:
		  gam1 = -xla - h1;
		  gam2 = -xla + h1;
		  gamm = gam1;
		  if (gam1*gam1 > gam2*gam2) { gamm=gam2; }
		  del1 = -xmu - h2;
		  del2 = -xmu + h2;
		  delm = del1;
		  if (del1*del1 > del2*del2) { delm = del2; }	

	//Now choose the line gamma=gamm and optimize delta:
		  gam = gamm;	  
		  delms = gam * u12;	  
		  aa = xmu + delms;	// look if delms is within [-xmu0+/-L/2]:
		  if (aa*aa <= h2*h2) {
		    del=delms;
		  }		// somewhere along the side gam=gamm
		  else {
	//delms out of range --> corner next to delms!
		    del = del1;
		    a1 = delms - del1;
		    a2 = delms - del2;
		    if (a1*a1 > a2*a2) {del = del2; }
		  }
		  
	// Distance at these gam, del:
		  f1 = gam*gam+del*del-2.*gam*del*u12;
		  gc1 = gam;
		  dc1 = del;
		  
	//Now choose the line delta=deltam and optimize gamma:
		  del=delm;	  
		  gamms=del*u12;	  
		  aa=xla+gamms;	// look if gamms is within [-xla0+/-L/2]:
		  if (aa*aa <= h1*h1) {
		    gam=gamms; }		// somewhere along the side gam=gamm
		  else {
	// gamms out of range --> corner next to gamms!
		    gam=gam1;
		    a1=gamms-gam1;
		    a2=gamms-gam2;
		    if (a1*a1 > a2*a2) gam=gam2;
		  }
	// Distance at these gam, del:
		  f2 = gam*gam+del*del-2.*gam*del*u12;
		  gc2 = gam;
		  dc2 = del;
		  
	// Compare f1 and f2 to find risq:
		  risq=f1;
		  answer.rix1 = x1 + (xla + gc1) * nx1;
		  answer.riy1 = y1 + (xla + gc1) * ny1;
		  answer.riz1 = z1 + (xla + gc1) * nz1;
		  answer.rix2 = x2 + (xmu + dc1) * nx2;
		  answer.riy2 = y2 + (xmu + dc1) * ny2;
		  answer.riz2 = z2 + (xmu + dc1) * nz2;
		  
		  if(f2 < f1) {
			  risq=f2;
			  answer.rix1 = x1 + (xla + gc2) * nx1;
			  answer.riy1 = y1 + (xla + gc2) * ny1;
			  answer.riz1 = z1 + (xla + gc2) * nz1;
			  answer.rix2 = x2 + (xmu + dc2) * nx2;
			  answer.riy2 = y2 + (xmu + dc2) * ny2;
			  answer.riz2 = z2 + (xmu + dc2) * nz2;
		  
		  }
	}
	
	return answer;
}

vector<Cell *> cells;

my6Vec cell_cell_gforce(int i, int j) {
	// This function returns the forces acting on cell "i" due to contact with cell "j"
	
	my6Vec F;
	
	overlapVec nij;
	nij = get_overlap_Struct(cells[i], cells[j]);
	
	double rix = cells[i]->get_x();
	double riy = cells[i]->get_y();
	double riz = cells[i]->get_z();
	double rinx = cells[i]->get_nx();
	double riny = cells[i]->get_ny();
	double rinz = cells[i]->get_nz();
	
	double rjx = cells[j]->get_x();
	double rjy = cells[j]->get_y();
	double rjz = cells[j]->get_z();
	double rjnx = cells[j]->get_nx();
	double rjny = cells[j]->get_ny();
	double rjnz = cells[j]->get_nz();
	
	double vx, vy, vz, wx, wy, wz;
  	vx = nij.rix1 - cells[i]->get_x();
  	vy = nij.riy1 - cells[i]->get_y();
  	vz = nij.riz1 - cells[i]->get_z();
  	wx = nij.rix2 - cells[j]->get_x();
  	wy = nij.riy2 - cells[j]->get_y();
  	wz = nij.riz2 - cells[j]->get_z();
	
	double check; // This constant determines the direction of the coordinate "s",
	// which gives the distance along the cell cylinder line from the cell center-of-mass.
	double ai = 1;
	double aj = 1;
	
	check = rinx * vx + riny * vy + rinz * vz;
	if (check < 0) {
		ai = -1;
	}
	
	check = rjnx * wx + rjny * wy + rjnz * wz;
	if (check < 0) {
		aj = -1;
	}
	
	// Intermediate values that are used to calculate the cell-to-cell contact force.
	double ac = sqrt(vx*vx + vy*vy + vz*vz);
	double bc = sqrt(wx*wx + wy*wy + wz*wz);
	
	double l = cells[i]->get_l();
	
	double qij2 = pow(ac*ai*rinx + rix - aj*bc*rjnx - rjx,2) + 
	   pow(ac*ai*riny + riy - aj*bc*rjny - rjy,2) + 
	   pow(ac*ai*rinz + riz - aj*bc*rjnz - rjz,2);
	
	double QT = (2*A_0)/sqrt(qij2) - 5*E_0*pow(-pow(qij2,0.16666666666666666) + 
      (2*R)/pow(qij2,0.3333333333333333),1.5);
	
	double a = -E_0;
	
	F.x = 0;
	F.y = 0;
	F.z = 0;
	F.nx = 0;
	F.ny = 0;
	F.nz = 0;
	
	double overlap = 2*R - sqrt( (nij.rix1 - nij.rix2)*(nij.rix1 - nij.rix2) +
			(nij.riy1 - nij.riy2)*(nij.riy1 - nij.riy2) +
						(nij.riz1 - nij.riz2)*(nij.riz1 - nij.riz2) ) ;
	
	// Check if the cells are overlapping and that their locations do not perfectly coincide
	// These checks are redudant due to small, numerical errors in the function that calculates overlap.
	if ( overlap > 0 and sqrt(qij2) > 0 and sqrt(qij2) < 2*R and pow(-sqrt(qij2) + 2*R,1.5) > 0) {
		F.x = (QT*(ac*ai*rinx + rix - aj*bc*rjnx - rjx))/2.;
		F.y = (QT*(ac*ai*riny + riy - aj*bc*rjny - rjy))/2.;
		F.z = (QT*(ac*ai*rinz + riz - aj*bc*rjnz - rjz))/2.;
		F.nx = (ac*ai*QT*(ac*ai*rinx + rix - aj*bc*rjnx - rjx))/2.;
		F.ny = (ac*ai*QT*(ac*ai*riny + riy - aj*bc*rjny - rjy))/2.;
		F.nz = (ac*ai*QT*(ac*ai*rinz + riz - aj*bc*rjnz - rjz))/2.;
	} else if ( sqrt( (nij.rix1 - nij.rix2)*(nij.rix1 - nij.rix2) +
			(nij.riy1 - nij.riy2)*(nij.riy1 - nij.riy2) +
						(nij.riz1 - nij.riz2)*(nij.riz1 - nij.riz2) ) == 0 ) {
		cout << "Error 1. cells are on top of each other in gforce." << endl;
	} else {
		//cout << "Error 2. cells are ? in gforce." << endl;
	}
	
	// Correct the sign.
	F.x = -F.x;
	F.y = -F.y;
	F.z = -F.z;
	F.nx = -F.nx;
	F.ny = -F.ny;
	F.nz = -F.nz;
	
	return F;
}

my6Vec cell_surface_gforce(Cell* cell_1) {
	// Return the surface torque acting on cell 1
	
	double t;
	
	double rix = cell_1->get_x();
	double riy = cell_1->get_y();
	double riz = cell_1->get_z();
	double rinx = cell_1->get_nx();
	double riny = cell_1->get_ny();
	double rinz = cell_1->get_nz();
	
	double nzi = cell_1->get_nz();
	
	
	double check = 1;
	
	double l = cell_1->get_l();
	double z = cell_1->get_z();
	
	if (rinz < 0) {
		rinz = -rinz;
		check = -1;
	}
	
	if (nzi < -1) {
		nzi = -1;
	}
	if (nzi > 1) {
		nzi = 1;
	}
	
	if (nzi > 0) {
		t = asin(nzi);
	} else {
		t = asin(-nzi);
	}
	
	if (rinz < -1) {
		rinz = -1;
	}
	if (rinz > 1) {
		rinz = 1;
	}
	
	double h1 = z + l*nzi/2.0;
	double h2 = z - l*nzi/2.0;
	
	
	if (h1 > h2) {
		h1 = h2;
	}
	
	
	my6Vec F;
	F.x = 0;
	F.y = 0;
	F.z = 0;
	F.nx = 0;
	F.ny = 0;
	F.nz = 0;
	
	if (h1 > R) {
		// No contact
	} else if (h1 > (R - l*sin(t))) {
		// Partial contact
		F.z -= (A_2*M_PI*rinz)/R - (4*E_2*sqrt(R)*rinz*pow(R + (l*rinz)/2. - riz,1.5))/3. - 
		   (E_1*(1 - pow(rinz,2))*pow(-R - (l*rinz)/2. + riz,2))/rinz + 
		   (A_1*(1 - pow(rinz,2))*sqrt(1 - (-(l*rinz)/2. + riz)/R))/(R*rinz);
		F.nz -= -(A_2*l*M_PI*rinz)/(2.*R) - (A_2*M_PI*(R + (l*rinz)/2. - riz))/R + 
		   (2*E_2*l*sqrt(R)*rinz*pow(R + (l*rinz)/2. - riz,1.5))/3. + 
		   (8*E_2*sqrt(R)*pow(R + (l*rinz)/2. - riz,2.5))/15. + 
		   (E_1*l*(1 - pow(rinz,2))*pow(-R - (l*rinz)/2. + riz,2))/(2.*rinz) + 
		   (2*E_1*pow(-R - (l*rinz)/2. + riz,3))/3. + 
		   (E_1*(1 - pow(rinz,2))*pow(-R - (l*rinz)/2. + riz,3))/(3.*pow(rinz,2)) - 
		   (A_1*l*(1 - pow(rinz,2))*sqrt(1 - (-(l*rinz)/2. + riz)/R))/(2.*R*rinz) + 
		   (4*A_1*pow(1 - (-(l*rinz)/2. + riz)/R,1.5))/3. + 
		   (2*A_1*(1 - pow(rinz,2))*pow(1 - (-(l*rinz)/2. + riz)/R,1.5))/(3.*pow(rinz,2));
		
	} else if (h1 > 0) {
		// Full contact
		if (abs(t) < 1e-6) {
			// linear version. Contact forces are linearized for small elevation angle
			// to avoid numerical instabilities
			F.z -= -2*E_1*l*R + (A_1*l)/(2.*pow(R,1.5)*sqrt(R - riz)) + 2*E_1*l*riz + 
		   pow(rinz,2)*(2*E_1*l*R + (A_1*pow(l,3))/(64.*pow(R,1.5)*pow(R - riz,2.5)) + 
		      (2*E_2*l*pow(R,2.5))/(3.*pow(R - riz,1.5)) - (A_1*l)/(2.*pow(R,1.5)*sqrt(R - riz)) - 
		      (8*E_2*l*pow(R,1.5))/(3.*sqrt(R - riz)) - 2*E_1*l*riz - 
		      (4*E_2*l*pow(R,1.5)*riz)/(3.*pow(R - riz,1.5)) + (8*E_2*l*sqrt(R)*riz)/(3.*sqrt(R - riz)) + 
		      (2*E_2*l*sqrt(R)*pow(riz,2))/(3.*pow(R - riz,1.5)));
			F.nz -= 2*rinz*((E_1*pow(l,3))/12. - (A_2*l*M_PI)/R - E_1*l*pow(R,2) + 
		     (A_1*pow(l,3))/(96.*pow(R,1.5)*pow(R - riz,1.5)) + (4*E_2*l*pow(R,2.5))/(3.*sqrt(R - riz)) + 
		     (A_1*l*sqrt(R - riz))/pow(R,1.5) + 2*E_1*l*R*riz - (8*E_2*l*pow(R,1.5)*riz)/(3.*sqrt(R - riz)) - 
		     E_1*l*pow(riz,2) + (4*E_2*l*sqrt(R)*pow(riz,2))/(3.*sqrt(R - riz)));
		} else {
			// nonlinear version
			F.z -= (E_1*l*(1 - pow(rinz,2))*(3*l*rinz + 6*(-R - (l*rinz)/2. + riz)))/3. - 
   (2*A_1*(1 - pow(rinz,2))*(-(l*rinz)/(2.*sqrt(R - (l*rinz)/2. - riz)) + sqrt(R - (l*rinz)/2. - riz) - 
        sqrt(R + (l*rinz)/2. - riz) - (1/(2.*sqrt(R - (l*rinz)/2. - riz)) - 
           1/(2.*sqrt(R + (l*rinz)/2. - riz)))*(-R - (l*rinz)/2. + riz)))/(3.*pow(R,1.5)*rinz) - 
   (8*E_2*sqrt(R)*rinz*(2*l*rinz*sqrt(R - (l*rinz)/2. - riz) - 
        2*(-sqrt(R - (l*rinz)/2. - riz) + sqrt(R + (l*rinz)/2. - riz))*(-R - (l*rinz)/2. + riz) - 
        (1/(2.*sqrt(R - (l*rinz)/2. - riz)) - 1/(2.*sqrt(R + (l*rinz)/2. - riz)))*
         pow(-R - (l*rinz)/2. + riz,2) - (l*rinz*(-2*R + l*rinz + 2*(-(l*rinz)/2. + riz)))/
         (2.*sqrt(R - (l*rinz)/2. - riz))))/15.;
			F.nz -= (-2*A_2*l*M_PI*rinz)/R - (2*A_1*(1 - pow(rinz,2))*
      ((l*(-sqrt(R - (l*rinz)/2. - riz) + sqrt(R + (l*rinz)/2. - riz)))/2. - 
        (pow(l,2)*rinz)/(4.*sqrt(R - (l*rinz)/2. - riz)) + l*sqrt(R - (l*rinz)/2. - riz) - 
        (l/(4.*sqrt(R - (l*rinz)/2. - riz)) + l/(4.*sqrt(R + (l*rinz)/2. - riz)))*(-R - (l*rinz)/2. + riz)))/
    (3.*pow(R,1.5)*rinz) + (4*A_1*(l*rinz*sqrt(R - (l*rinz)/2. - riz) - 
        (-sqrt(R - (l*rinz)/2. - riz) + sqrt(R + (l*rinz)/2. - riz))*(-R - (l*rinz)/2. + riz)))/
    (3.*pow(R,1.5)) + (2*A_1*(1 - pow(rinz,2))*
      (l*rinz*sqrt(R - (l*rinz)/2. - riz) - (-sqrt(R - (l*rinz)/2. - riz) + sqrt(R + (l*rinz)/2. - riz))*
         (-R - (l*rinz)/2. + riz)))/(3.*pow(R,1.5)*pow(rinz,2)) - 
   (8*E_2*sqrt(R)*rinz*(l*(-sqrt(R - (l*rinz)/2. - riz) + sqrt(R + (l*rinz)/2. - riz))*
         (-R - (l*rinz)/2. + riz) - (l/(4.*sqrt(R - (l*rinz)/2. - riz)) + l/(4.*sqrt(R + (l*rinz)/2. - riz)))*
         pow(-R - (l*rinz)/2. + riz,2) - (pow(l,2)*rinz*(-2*R + l*rinz + 2*(-(l*rinz)/2. + riz)))/
         (4.*sqrt(R - (l*rinz)/2. - riz)) + l*sqrt(R - (l*rinz)/2. - riz)*
         (-2*R + l*rinz + 2*(-(l*rinz)/2. + riz))))/15. - 
   (8*E_2*sqrt(R)*(-((-sqrt(R - (l*rinz)/2. - riz) + sqrt(R + (l*rinz)/2. - riz))*
           pow(-R - (l*rinz)/2. + riz,2)) + 
        l*rinz*sqrt(R - (l*rinz)/2. - riz)*(-2*R + l*rinz + 2*(-(l*rinz)/2. + riz))))/15. + 
   (E_1*l*(1 - pow(rinz,2))*(-(pow(l,2)*rinz)/2. - 3*l*(-R - (l*rinz)/2. + riz) + 
        l*(-3*R + l*rinz + 3*(-(l*rinz)/2. + riz))))/3. - 
   (2*E_1*l*rinz*(3*pow(-R - (l*rinz)/2. + riz,2) + l*rinz*(-3*R + l*rinz + 3*(-(l*rinz)/2. + riz))))/3.;
		}
		
	} else {
		cout << "ERROR: Torque is below the ground!" << endl;
		return F;
	}
	
	F.nz = check * F.nz;
	
	return F;
	
}

int below_surface(Cell* cell_1) {
	// Determine if the cell is below the ground
	
	double t;
	
	double nzi = cell_1->get_nz();
	if (nzi < -1) {
		nzi = -1;
	}
	if (nzi > 1) {
		nzi = 1;
	}
	
	if (nzi > 0) {
		t = asin(nzi);
	} else {
		t = asin(-nzi);
	}
	
	if (t == 0) {
		t = 1e-10;
	}
	
	double l = cell_1->get_l();
	double z = cell_1->get_z();
	
	double h1 = z + l*nzi/2.0;
	double h2 = z - l*nzi/2.0;
	
	if (h1 > h2) {
		h1 = h2;
	}
	
	if (h1 > R) {
		// No contact
		return 1;
	} else if (h1 > (R - l*sin(t))) {
		// Partial contact
		return 1;
	} else if (h1 > 0) {
		// Full contact
		return 1;
	} else {
		// Below ground
		return 0;
	}
}

int partial_contact(Cell* cell_1) {
	// Determine if the cell is reorienting
	
	double t;
	
	double nzi = cell_1->get_nz();
	if (nzi < -1) {
		nzi = -1;
	}
	if (nzi > 1) {
		nzi = 1;
	}
	
	if (nzi > 0) {
		t = asin(nzi);
	} else {
		t = asin(-nzi);
	}
	
	if (t == 0) {
		t = 1e-10;
	}
	
	double l = cell_1->get_l();
	double z = cell_1->get_z();
	
	double h1 = z + l*nzi/2.0;
	double h2 = z - l*nzi/2.0;
	
	if (h1 > h2) {
		h1 = h2;
	}
	
	if (h1 > R) {
		// No contact
		return 1;
	} else if (h1 > (R - l*sin(t))) {
		// Partial contact
		return 1;
	} else if (h1 > 0) {
		// Full contact
		return 0;
	} else {
		// Below ground
		return 0;
	}
}

int cell_contact(Cell* cell_1) {
	// Determine if the cell is reorienting
	
	double t;
	
	double nzi = cell_1->get_nz();
	if (nzi < -1) {
		nzi = -1;
	}
	if (nzi > 1) {
		nzi = 1;
	}
	
	if (nzi > 0) {
		t = asin(nzi);
	} else {
		t = asin(-nzi);
	}
	
	if (t == 0) {
		t = 1e-10;
	}
	
	double l = cell_1->get_l();
	double z = cell_1->get_z();
	
	double h1 = z + l*nzi/2.0;
	double h2 = z - l*nzi/2.0;
	
	if (h1 > h2) {
		h1 = h2;
	}
	
	//cout << "h1: " << h1 << endl;
	
	if (h1 > R) {
		//cout << "h1 - R: " << h1 - R << endl;
		// No contact
		return 0;
	} else if (h1 > (R - l*sin(t))) {
		// Partial contact
		return 1;
	} else if (h1 > 0) {
		// Full contact
		return 1;
	} else {
		// Below ground
		cout << "Error, below ground in cell contact?" << endl;
		return 0;
		
	}
}

double cell_h(Cell* cell_1) {
	// Determine if the cell is reorienting
	
	double t;
	
	double nzi = cell_1->get_nz();
	if (nzi < -1) {
		nzi = -1;
	}
	if (nzi > 1) {
		nzi = 1;
	}
	
	if (nzi > 0) {
		t = asin(nzi);
	} else {
		t = asin(-nzi);
	}
	
	if (t == 0) {
		t = 1e-10;
	}
	
	double l = cell_1->get_l();
	double z = cell_1->get_z();
	
	double h1 = z + l*nzi/2.0;
	double h2 = z - l*nzi/2.0;
	
	if (h1 > h2) {
		h1 = h2;
	}
	
	//cout << "h1: " << h1 << endl;
	
	return h1;
	
}

my6Vec net_gforce_elon(int i) {
	// Returns the net force on each generalized coordinate of cell i from cell-cell and cell-surface interactions
	
	my6Vec FF, Fi;
	FF.x = 0;
	FF.y = 0;
	FF.z = 0;
	FF.nx = 0;
	FF.ny = 0;
	FF.nz = 0;
	
	for (int j = 0; j != cells.size(); j++) {
		if (i != j) {
			Fi = cell_cell_gforce(i, j);
			
			FF.x += Fi.x;
			FF.y += Fi.y;
			FF.z += Fi.z;
			FF.nx += Fi.nx;
			FF.ny += Fi.ny;
			FF.nz += Fi.nz;
		}
	}
	
	// Calculate surface force:
	Fi = cell_surface_gforce(cells[i]);
	
	FF.z += Fi.z;
	FF.nz += Fi.nz;
		
	if (xy == 1) {
		FF.z = 0;
		FF.nz = 0;
	}
	if (xz == 1) {
		FF.y = 0;
		FF.ny = 0;
	}
	
	return FF;
}

double net_gforce_mag(int i) {
	// Returns the net force on each generalized coordinate of cell i from cell-cell and cell-surface interactions
	
	my6Vec FF, Fi;
	FF.x = 0;
	FF.y = 0;
	FF.z = 0;
	FF.nx = 0;
	FF.ny = 0;
	FF.nz = 0;
	
	double fmag = 0;
	
	for (int j = 0; j != cells.size(); j++) {
		if (i != j) {
			Fi = cell_cell_gforce(i, j);
			
			FF.x += Fi.x * Fi.x;
			FF.y += Fi.y * Fi.y;
			
			fmag += sqrt(Fi.x * Fi.x + Fi.y * Fi.y);
			//FF.z += Fi.z;
			//FF.nx += Fi.nx;
			//FF.ny += Fi.ny;
			//FF.nz += Fi.nz;
		}
	}
	
	// Calculate surface force:
	//Fi = cell_surface_gforce(cells[i]);
	
	//FF.z += Fi.z;
	//FF.nz += Fi.nz;
	
	return fmag;
}

derivFunc get_derivs(double nxi, double nyi, double nzi, double l, double z, my6Vec F) {
	// This function returns the rate of change of the cell coordinates, i.e. dq_i / dt
	// The rate of change depends on the forces as well as the friction
	
	derivFunc df;
	df.xd = 0;
	df.yd = 0;
	df.zd = 0;
	df.nxd = 0;
	df.nyd = 0;
	
	double Fx = F.x;
	double Fy = F.y;
	double Fz = F.z;
	double Tx = F.nx;
	double Ty = F.ny;
	double Tz = F.nz;
	
	double t;
	
	if (nzi < -1) {
		nzi = -1;
	}
	if (nzi > 1) {
		nzi = 1;
	}
	
	int trip = 0;
	
	if (nzi > 0) {
		t = asin(nzi);
	} else {
		t = asin(-nzi);
		nxi = -nxi;
		nyi = -nyi;
		nzi = -nzi;
	}
	
	double h1 = z + l*nzi/2.0;
	double h2 = z - l*nzi/2.0;
	
	if (h1 > h2) {
		h1 = h2;
	}
	
	// Area constants
	double B0, B1, B2;
	
	if (h1 > R or 2*R - 2*z - l*sin(t) < 0) {
		// No contact
		B0 = 0;
		B1 = 0;
		B2 = 0;
	} else if (h1 > (R - l*sin(t))) {
		// Partial contact
		B0 = (2*(pow(nxi,2) + pow(nyi,2))*pow(1 - h1/R,1.5))/(3.*nzi) + (nzi*M_PI*(-h1 + R))/R;
		B1 = -(h1*M_PI) - (l*nzi*M_PI)/2. + (pow(h1,2)*M_PI)/(2.*R) + (h1*l*nzi*M_PI)/(2.*R) + (M_PI*R)/2. + 
   ((pow(nxi,2) + pow(nyi,2))*pow(1 - h1/R,1.5)*(-4*h1 - 5*l*nzi + 4*R))/(15.*pow(nzi,2));
		B2 = -(M_PI*(h1 - R)*(4*pow(h1,2) + 6*h1*l*nzi + 3*pow(l,2)*pow(nzi,2) - 8*h1*R - 6*l*nzi*R + 4*pow(R,2)))/(12.*nzi*R) + 
   ((pow(nxi,2) + pow(nyi,2))*pow(1 - h1/R,1.5)*(32*pow(h1,2) + 56*h1*l*nzi + 35*pow(l,2)*pow(nzi,2) - 8*(8*h1 + 7*l*nzi)*R + 
        32*pow(R,2)))/(210.*pow(nzi,3));
	} else if (h1 > 0) {
		// Full contact
		if (abs(t) < 1e-6) {
			// linearized version
			B0 = l*(pow(nxi,2) + pow(nyi,2))*sqrt(R)*sqrt(R - z);
			B1 = -(pow(l,3)*(pow(nxi,2) + pow(nyi,2))*nzi*sqrt(R))/(24.*sqrt(R - z));
			B2 = (pow(l,3)*(pow(nxi,2) + pow(nyi,2))*sqrt(R)*sqrt(R - z))/12.;
		} else {
			// Nonlinear
			B0 = l*pow(nzi,2)*M_PI*R + ((pow(nxi,2) + pow(nyi,2))*sqrt(R)*
      (l*nzi*(sqrt(-(l*nzi) + 2*R - 2*z) + sqrt(l*nzi + 2*R - 2*z)) - 2*(sqrt(-(l*nzi) + 2*R - 2*z) - sqrt(l*nzi + 2*R - 2*z))*(R - z)))/
    (3.*sqrt(2)*nzi);
			B1 = ((pow(nxi,2) + pow(nyi,2))*sqrt(R)*(3*pow(l,2)*pow(nzi,2)*(sqrt(-(l*nzi) + 2*R - 2*z) - sqrt(l*nzi + 2*R - 2*z)) - 
       2*l*nzi*(sqrt(-(l*nzi) + 2*R - 2*z) + sqrt(l*nzi + 2*R - 2*z))*(R - z) - 
       8*(sqrt(-(l*nzi) + 2*R - 2*z) - sqrt(l*nzi + 2*R - 2*z))*pow(R - z,2)))/(30.*sqrt(2)*pow(nzi,2));
			B2 = (pow(l,3)*pow(nzi,2)*M_PI*R)/12. + ((pow(nxi,2) + pow(nyi,2))*sqrt(R)*
      (15*pow(l,3)*pow(nzi,3)*(sqrt(-(l*nzi) + 2*R - 2*z) + sqrt(l*nzi + 2*R - 2*z)) - 
        6*pow(l,2)*pow(nzi,2)*(sqrt(-(l*nzi) + 2*R - 2*z) - sqrt(l*nzi + 2*R - 2*z))*(R - z) - 
        16*l*nzi*(sqrt(-(l*nzi) + 2*R - 2*z) + sqrt(l*nzi + 2*R - 2*z))*pow(R - z,2) - 
        64*(sqrt(-(l*nzi) + 2*R - 2*z) - sqrt(l*nzi + 2*R - 2*z))*pow(R - z,3)))/(420.*sqrt(2)*pow(nzi,3));
		}
	
	} else {
		cout << "ERROR: Cell is below the ground!" << endl;
		//return answer;
	}
	
	df.xd = (12*(B1*NU_1*(pow(l,3)*nxi*nyi*NU_0*(l*(-(nyi*Tx) + nxi*Ty)*NU_0 - (B1*Fy*nxi + B0*nyi*Tx - B0*nxi*Ty)*NU_1) + 
          12*nxi*nzi*(nzi*Tx - nxi*Tz)*(pow(B1,2)*pow(NU_1,2) - (l*NU_0 + B0*NU_1)*((pow(l,3)*NU_0)/12. + B2*NU_1))) + 
       Fx*(pow(l,3)*nxi*pow(nyi,2)*NU_0*(l*NU_0 + B0*NU_1)*((pow(l,3)*NU_0)/12. + B2*NU_1) - 
          nxi*(pow(l,3)*(pow(nxi,2) + pow(nzi,2))*NU_0 + 12*B2*pow(nzi,2)*NU_1)*
           (pow(B1,2)*pow(NU_1,2) - (l*NU_0 + B0*NU_1)*((pow(l,3)*NU_0)/12. + B2*NU_1)))))/
   (nxi*(pow(l,4)*pow(NU_0,2) + 12*B2*l*NU_0*NU_1 + B0*pow(l,3)*NU_0*NU_1 - 12*(pow(B1,2) - B0*B2)*pow(NU_1,2))*
     (pow(l,4)*(pow(nxi,2) + pow(nyi,2) + pow(nzi,2))*pow(NU_0,2) + 12*B2*l*pow(nzi,2)*NU_0*NU_1 + 
       B0*pow(l,3)*(pow(nxi,2) + pow(nyi,2) + pow(nzi,2))*NU_0*NU_1 - 12*(pow(B1,2) - B0*B2)*pow(nzi,2)*pow(NU_1,2)));
	df.yd = (-12*B1*NU_1*(pow(l,4)*(-(nxi*nyi*Tx) + pow(nxi,2)*Ty + nzi*(nzi*Ty - nyi*Tz))*pow(NU_0,2) + 12*B2*l*nzi*(nzi*Ty - nyi*Tz)*NU_0*NU_1 + 
        pow(l,3)*(B1*Fx*nxi*nyi + B0*(-(nxi*nyi*Tx) + pow(nxi,2)*Ty + nzi*(nzi*Ty - nyi*Tz)))*NU_0*NU_1 + 
        12*(pow(B1,2) - B0*B2)*nzi*(-(nzi*Ty) + nyi*Tz)*pow(NU_1,2)) + 
     Fy*(pow(l,7)*pow(NU_0,3) + B0*pow(l,6)*pow(NU_0,2)*NU_1 + 
        12*B2*pow(l,4)*(pow(nxi,2) + pow(nyi,2) + 2*pow(nzi,2))*pow(NU_0,2)*NU_1 + 
        144*pow(B2,2)*l*pow(nzi,2)*NU_0*pow(NU_1,2) + 
        12*pow(l,3)*(-(pow(B1,2)*(pow(nyi,2) + pow(nzi,2))) + B0*B2*(pow(nxi,2) + pow(nyi,2) + 2*pow(nzi,2)))*NU_0*
         pow(NU_1,2) + 144*B2*(-pow(B1,2) + B0*B2)*pow(nzi,2)*pow(NU_1,3)))/
   ((pow(l,4)*pow(NU_0,2) + 12*B2*l*NU_0*NU_1 + B0*pow(l,3)*NU_0*NU_1 - 12*(pow(B1,2) - B0*B2)*pow(NU_1,2))*
     (pow(l,4)*pow(NU_0,2) + B0*pow(l,3)*NU_0*NU_1 + 12*B2*l*pow(nzi,2)*NU_0*NU_1 - 12*(pow(B1,2) - B0*B2)*pow(nzi,2)*pow(NU_1,2)));
	df.zd = Fz/(l*NU_0);
	df.nxd = (12*(pow(l,5)*(pow(nyi,2)*Tx - nxi*nyi*Ty + nzi*(nzi*Tx - nxi*Tz))*pow(NU_0,3) + 
       12*B2*pow(l,2)*nzi*(nzi*Tx - nxi*Tz)*pow(NU_0,2)*NU_1 + 
       pow(l,4)*(B1*(Fy*nxi*nyi - Fx*(pow(nyi,2) + pow(nzi,2))) + 2*B0*(pow(nyi,2)*Tx - nxi*nyi*Ty + nzi*(nzi*Tx - nxi*Tz)))*
        pow(NU_0,2)*NU_1 + 12*l*nzi*(-(nzi*(B1*B2*Fx + pow(B1,2)*Tx - 2*B0*B2*Tx)) + (pow(B1,2) - 2*B0*B2)*nxi*Tz)*NU_0*pow(NU_1,2) - 
       B0*pow(l,3)*(B1*(-(Fy*nxi*nyi) + Fx*(pow(nyi,2) + pow(nzi,2))) + 
          B0*(-(pow(nyi,2)*Tx) + nxi*nyi*Ty + nzi*(-(nzi*Tx) + nxi*Tz)))*NU_0*pow(NU_1,2) + 
       12*(pow(B1,2) - B0*B2)*nzi*(B1*Fx*nzi - B0*nzi*Tx + B0*nxi*Tz)*pow(NU_1,3)))/
   ((pow(l,4)*pow(NU_0,2) + 12*B2*l*NU_0*NU_1 + B0*pow(l,3)*NU_0*NU_1 - 12*(pow(B1,2) - B0*B2)*pow(NU_1,2))*
     (pow(l,4)*pow(NU_0,2) + B0*pow(l,3)*NU_0*NU_1 + 12*B2*l*pow(nzi,2)*NU_0*NU_1 - 12*(pow(B1,2) - B0*B2)*pow(nzi,2)*pow(NU_1,2)));
	df.nyd = (12*(pow(l,5)*(-(nxi*nyi*Tx) + pow(nxi,2)*Ty + nzi*(nzi*Ty - nyi*Tz))*pow(NU_0,3) + 
       12*B2*pow(l,2)*nzi*(nzi*Ty - nyi*Tz)*pow(NU_0,2)*NU_1 - 
       pow(l,4)*(B1*(-(Fx*nxi*nyi) + Fy*(pow(nxi,2) + pow(nzi,2))) + 2*B0*(nxi*nyi*Tx - pow(nxi,2)*Ty + nzi*(-(nzi*Ty) + nyi*Tz)))*
        pow(NU_0,2)*NU_1 + 12*l*nzi*(-(nzi*(B1*B2*Fy + pow(B1,2)*Ty - 2*B0*B2*Ty)) + (pow(B1,2) - 2*B0*B2)*nyi*Tz)*NU_0*pow(NU_1,2) + 
       B0*pow(l,3)*(B1*(Fx*nxi*nyi - Fy*(pow(nxi,2) + pow(nzi,2))) + B0*(-(nxi*nyi*Tx) + pow(nxi,2)*Ty + nzi*(nzi*Ty - nyi*Tz)))*NU_0*
        pow(NU_1,2) + 12*(pow(B1,2) - B0*B2)*nzi*(B1*Fy*nzi - B0*nzi*Ty + B0*nyi*Tz)*pow(NU_1,3)))/
   ((pow(l,4)*pow(NU_0,2) + 12*B2*l*NU_0*NU_1 + B0*pow(l,3)*NU_0*NU_1 - 12*(pow(B1,2) - B0*B2)*pow(NU_1,2))*
     (pow(l,4)*pow(NU_0,2) + B0*pow(l,3)*NU_0*NU_1 + 12*B2*l*pow(nzi,2)*NU_0*NU_1 - 12*(pow(B1,2) - B0*B2)*pow(nzi,2)*pow(NU_1,2)));
	
	df.nzd = (-12*(l*(nxi*nzi*Tx + nyi*nzi*Ty + (-1 + pow(nzi,2))*Tz)*NU_0 + 
       (-(B1*(Fx*nxi + Fy*nyi)*nzi) + B0*(nxi*nzi*Tx + nyi*nzi*Ty + (-1 + pow(nzi,2))*Tz))*NU_1))/
   (pow(l,4)*pow(NU_0,2) + B0*pow(l,3)*NU_0*NU_1 + 12*B2*l*pow(nzi,2)*NU_0*NU_1 - 12*(pow(B1,2) - B0*B2)*pow(nzi,2)*pow(NU_1,2));
	
	return df;
}

int grow_func (double t, const double y[], double f[], void *params) {
  (void)(t);
  
  // Calculate the rhs of the differential equation, dy/dt = ?
  
  // The degrees of freedom are as follows:
  // f[0] = cell 1, x pos
  // f[1] = cell 1, y pos
  // f[2] = cell 1, z pos
  // f[3] = cell 1, nx
  // f[4] = cell 1, ny
  // f[5] = cell 1, nz
  // f[6] = length
  // f[7] cell 2, x pos...
  
  double tx, ty, tz, t_net, nx1, ny1, nz1, nx2, ny2, nz2, l, d, tnz, tn;
  double dx, dy, dz;
  
  double norm;
  
  double ai;
  
  my6Vec F, noise;
  
  myVec Fi, Ti;
  
  derivFunc df, df2;
  
  double fx, fy, fz;
  double wx, wy, wz;
  
  int confine;
  
  // Calculate the force on cell i
  int dof = 0;
  int i = 0;
  while (i < cells.size()) {  
	  if (i >= cells.size()) {
	  	  f[dof] = 0;
		  dof = dof + 1;
		  
		  
	  }
	  else {
		  
		  nx1 = y[dof+3];
		  ny1 = y[dof+4];
		  nz1 = y[dof+5];
		  norm = vnorm(nx1, ny1, nz1);
		  nx1 = nx1/norm;
		  ny1 = ny1/norm;
		  nz1 = nz1/norm;
		  
		  l = y[dof+6];
		  
		  cells[i]->set_pos(y[dof], y[dof+1], y[dof+2]);
		  cells[i]->set_n(nx1, ny1, nz1);
		  cells[i]->set_l(l);
		  
		  i++;
		  dof = dof + DOF_GROW;
	  }
  }
  
  // Calculate the force on cell i
  dof = 0;
  i = 0;
  while (i < cells.size()) {  
	  if (i >= cells.size()) {
	  	  f[dof] = 0;
		  dof = dof + 1;
	  }
	  else {
		  
		  nx1 = y[dof+3];
		  ny1 = y[dof+4];
		  nz1 = y[dof+5];
		  norm = vnorm(nx1, ny1, nz1);
		  nx1 = nx1/norm;
		  ny1 = ny1/norm;
		  nz1 = nz1/norm;
		  
		  l = y[dof+6];
		  
		  confine = cells[i]->get_confine();
		  
		  if (below_surface(cells[i]) == 0) {
			  return GSL_EBADFUNC;
		  }
		  
		  F = net_gforce_elon(i);
		  noise = cells[i]->get_noise();
		  
		  dx = noise.x;
		  dy = noise.y;
		  dz = noise.z;
		  dx = 0;
		  dy = 0;
		  dz = 0;
		  
	  	  if (xy == 1 or confine == 1) {
	  		  dz = 0;
			  F.z = 0;
	  	  }
	  	  if (xz == 1) {
	  	  	  dy = 0;
			  F.y = 0;
	  	  }
		  
		  F.x = F.x + dx;
		  F.y = F.y + dy;
		  F.z = F.z + dz;
		  
		  dx = noise.nx;
		  dy = noise.ny;
		  dz = noise.nz;
		  
		  if (xy == 1 or confine == 1) {
			  dz = 0;
			  F.nz = 0;
	  	  }
	  	  if (xz == 1) {
	  	  	  dy = 0;
			  F.ny = 0;
	  	  }
		  
		  F.nx = F.nx + dx;
		  F.ny = F.ny + dy;
		  F.nz = F.nz + dz;
		  
		  df = get_derivs(nx1, ny1, nz1, l, y[dof+2], F);
		  
		  
		  f[dof+0] = df.xd;
		  f[dof+1] = df.yd;
		  f[dof+2] = df.zd;
		  f[dof+3] = df.nxd;
		  f[dof+4] = df.nyd;
		  f[dof+5] = df.nzd;
		  
		  ai = cells[i]->get_ai();
		  
		  f[dof+6] = ai * (l + (4*R)/3.);
		  
		  i++;
		  dof = dof + DOF_GROW;
	  }
  }
  
  return GSL_SUCCESS;

}

double simple_grow(double tf) {
	// Evolve the system as each cell grows by elongation.
	double y[MAX_DOF];
	
	cout << "Growing" << endl;
	
	// Initialize output files
	ofstream myfile2;
	string myfile_name2 = my_name+"-vertical.txt";
	myfile2.open(myfile_name2);
	
	ofstream myfile;
	string my_mono_name = my_name+"-line.txt";
	myfile.open(my_mono_name);
	
	ofstream myfile3;
	string my_mono_name2 = my_name+"-force.txt";
	myfile3.open(my_mono_name2);
	
	ofstream myfile4;
	string my_mono_name3 = my_name+"-divide_pop.txt";
	myfile4.open(my_mono_name3);
	
	ofstream myfile5;
	string my_mono_name4 = my_name+"-time_lived.txt";
	myfile5.open(my_mono_name4);
	
	ofstream myfile6;
	string my_mono_name6 = my_name+"-upheaval.txt";
	myfile6.open(my_mono_name6);
	
	ofstream myfile7;
	string my_mono_name7 = my_name+"-division.txt";
	myfile7.open(my_mono_name7);
	unsigned long dim = MAX_DOF;
	
	// Different choices of the integration routine.
	//const gsl_odeiv_step_type * TT = gsl_odeiv_step_rk8pd;
	//const gsl_odeiv_step_type * TT = gsl_odeiv_step_rk4;
	const gsl_odeiv_step_type * TT = gsl_odeiv_step_rkf45;
	
	gsl_odeiv_step * s = gsl_odeiv_step_alloc (TT, dim);
	gsl_odeiv_control * c = gsl_odeiv_control_y_new (h0, 0);
	gsl_odeiv_evolve * e = gsl_odeiv_evolve_alloc (dim);
	
	gsl_odeiv_system sys = {grow_func, NULL, dim, NULL};
	
	double l_m, l_d, dx, dy, dz;
	int confine;
	
	int reset = 1;
  	int dof;
	int k;
	
	double t = 0.0, tcurrent = tstep;
	double h = h0;
	
	int alpha_trip = 0;
	
	double fmag;
	my6Vec F;
	double tnz, tn;
	
	double nnorm;
	
	double nz_0, nz_1, time_0;
	int sa0, sa1;
	
	double fm1, fm2, fm3;
	
	// Evolve the differential equation
	while (t < tf) {
		
		cout << t << endl;
		 
		// Evolve until time step to output data
		while (t < tcurrent) {
			
   		    if (reset == 1) {
				// Set initial conditions
				dof = 0;
				k = 0;
				
				while (k < cells.size()) {
					if (k >= cells.size()) {
						y[dof] = 0;
						dof = dof + 1;
					} else {
				  		y[dof+0] = cells[k]->get_x();
				  		y[dof+1] = cells[k]->get_y();
				  		y[dof+2] = cells[k]->get_z();
				  		y[dof+3] = cells[k]->get_nx();
				  		y[dof+4] = cells[k]->get_ny();
				  		y[dof+5] = cells[k]->get_nz();
				  		y[dof+6] = cells[k]->get_l();
						
						
						k++;
				  		dof = dof + DOF_GROW;
					}
					
			  	}
				
   		    }
			
			reset = 0;
			
			// Evolve the system by an adaptive step size, initial choice "h"
			int status = gsl_odeiv_evolve_apply (e, c, s, &sys, &t, tcurrent, &h, y);
			
			//cout << t << endl;
			
			// Set the coordinates after each step
	  		dof = 0;
			fmag = 0;
	  		for (int j = 0; j != cells.size(); j++) {
				
				cells[j]->set_noise();
				
	  		    if (xz == 1) {
	  			    y[dof+4] = 0;
	  		    }
		  	  
			  	
				if (xy == 1) {
					y[dof+5] = 0;
				}
				
				nnorm = vnorm(y[dof+3],y[dof+4],y[dof+5]);
				
				y[dof+3] = y[dof+3]/nnorm;
				y[dof+4] = y[dof+4]/nnorm;
				y[dof+5] = y[dof+5]/nnorm;
				
				cells[j]->set_pos(y[dof], y[dof+1], y[dof+2]);
				cells[j]->set_n(y[dof+3], y[dof+4], y[dof+5]);
				cells[j]->set_l(y[dof+6]);
				
				nz_0 = cells[j]->get_current_nz();
				sa0 = cells[j]->get_current_sa();
				
				nz_1 = cells[j]->get_nz();
				sa1 = cell_contact(cells[j]);
				
				cells[j]->set_current_nz(y[dof+5]);
				cells[j]->set_current_sa(sa1);
				
				dof = dof + DOF_GROW;
	  		}
			
			if (status != GSL_SUCCESS) {
				reset = 1;
				
				gsl_odeiv_step_reset(s);
				gsl_odeiv_evolve_reset (e);
				h = h/2.;
			} else {
				//reset = 0;
				k = 0;
				dof = 0;
				
				// Divide cells that are long enough (l_i > l_f)
				while (k < cells.size()) {
					if (y[dof+6] > cells[k]->get_Lf()) {
			  		    
						l_m = y[dof+6];
						l_d = (l_m + 2*R)/2.0 - 2*R;
						
			  		    if (xz == 1) {
			  			    y[dof+4] = 0;
			  		    }
						
						if (xy == 1) {
							y[dof+5] = 0;
						}
						
						dx = ((l_m + 2*R) / 4.0) * y[dof+3];
						dy = ((l_m + 2*R) / 4.0) * y[dof+4];
						dz = ((l_m + 2*R) / 4.0) * y[dof+5];
						
						confine = cells[k]->get_confine();
						
						Cell* c1 = new Cell(y[dof]+dx, y[dof+1]+dy, y[dof+2]+dz, y[dof+3], y[dof+4], y[dof+5], l_d);
						Cell* c2 = new Cell(y[dof]-dx, y[dof+1]-dy, y[dof+2]-dz, y[dof+3], y[dof+4], y[dof+5], l_d);
						
						c1->set_tborn(t);
						c2->set_tborn(t);
						
						c1->set_lin(cells[k]->get_lin());
						c1->set_lin("0");
						c2->set_lin(cells[k]->get_lin());
						c2->set_lin("1");
						
						if (confine == 1) {
							c1->set_confine();
							c2->set_confine();
						}
						
						c1->set_current_nz(y[dof+5]);
						c1->set_current_sa( cell_contact(c1) );
						c2->set_current_nz(y[dof+5]);
						c2->set_current_sa( cell_contact(c2) );
						
						delete cells[k];
						cells.erase(cells.begin()+k, cells.begin()+k+1);
						
						int kk = 0;
						if ( cell_contact(c1) == 1 ) {
							cells.insert(cells.begin()+k, c1);
							kk = kk + 1;
							
						} else {
							delete c1;
						}
						if ( cell_contact(c2) == 1 ) {
							cells.insert(cells.begin()+k, c2);
							
							kk = kk + 1;
						} else {

							delete c2;
						}
						
						k = k - 1 + kk;
						
						
						gsl_odeiv_step_reset(s);
						gsl_odeiv_evolve_reset (e);
						
						gsl_odeiv_evolve_free (e);
						gsl_odeiv_control_free (c);
						gsl_odeiv_step_free (s);
						
						h = h0;
						
						s = gsl_odeiv_step_alloc (TT, dim);
						c = gsl_odeiv_control_y_new (h0, 0);
						e = gsl_odeiv_evolve_alloc (dim);
						
						gsl_odeiv_system sys = {grow_func, NULL, dim, NULL};
						
						reset = 1;
					}
					k++;
			  		dof = dof + DOF_GROW;
				}
			}
		 }
		 
		 
		 // Output the data
		 int numcell = cells.size();
		 myfile << t << " " << numcell << endl;
		 
		 dof = 0;
		 double nz2 = 0;
		 double nz_sum = 0;
		 
		 int v_sum = 0;
		 my6Vec FF;
		 
		 for (int j = 0; j != cells.size(); j++) {
			 
			 FF = net_gforce_elon(j);
			 myfile << cells[j]->get_x() << " " << cells[j]->get_y() << " " << cells[j]->get_z() << " "
				 	<< cells[j]->get_nx() << " " << cells[j]->get_ny() << " " << cells[j]->get_nz() << " "
					<< cells[j]->get_l() << " " << cells[j]->get_ai() << " " << cell_contact(cells[j]) << " " << net_gforce_mag(j) << " " << FF.x << " " << FF.y << " " << cells[j]->get_lin() << endl;
			 dof = dof + DOF_GROW;
			 
		 }
		 
		 
		 tcurrent = tcurrent + tstep;
    }
	
	gsl_odeiv_evolve_free (e);
	gsl_odeiv_control_free (c);
	gsl_odeiv_step_free (s);
	
	return t;
}

int main(int argc, char * argv[]) {
	cout << "\n\n\n\n\n\n\n";
	cout << "Program running\n";
	
	// The program input.
	int trial;
	srand ((trial+1)*time(NULL));
	trial = atoi(argv[1]);  //Optional input for seed
	string my_index = argv[1];
	Lf = atof(argv[2])/10.0;
	E_1 = atof(argv[3]);
	NU_1 = atof(argv[4]) * (E_0) / 10.0; // surface drag coefficient
	
	// The ambient drag is chosen to be a fixed ratio of the surface drag.
	NU_0 = NU_1 / 100; // ambient (positional) drag
	
	
	string my_output = "output";
	
	
	my_output = my_output + "/" + my_index;
	mkdir(my_output.c_str(), 0700);
	
	my_output = my_output + "/" + argv[2];
	mkdir(my_output.c_str(), 0700);
	
	my_output = my_output + "/" + argv[3];
	mkdir(my_output.c_str(), 0700);
	
	my_output = my_output + "/" + argv[4];
	mkdir(my_output.c_str(), 0700);
	
	
	my_name = my_output + "/pack";
	
	cout << my_name << endl;
	
	
	// Initializations:
	cells.clear();
	cells.reserve(10000);
	
	
	// Create first cell:
	double ti = 0.0; // initial angle
	double nxi = cos(ti);
	double nzi = sin(ti);
	double nyi = 0;
	
	double x, y;
	double init_l = L0;
	
	double init_z = -pow(A_1,0.6666666666666666)/
   (2.*pow(2,0.3333333333333333)*pow(E_1,0.6666666666666666)) + R + nzi*init_l/2.0;
	
	if (E_1 == 0) {
		init_z = R;
	}
	
	if (xy == 1) { nzi = 0; }
	if (xz == 1) { nyi = 0; }
	double nm = sqrt(nxi*nxi + nyi*nyi + nzi*nzi);
	
	ofstream myfile;
	
	string my_mono_name = my_name+"-line.txt";
	myfile.open(my_mono_name);
	
	cout << "Declaring new cell" << endl;
	
	Cell * mb = new Cell(0, 0, init_z, nxi/nm, nyi/nm, nzi/nm, init_l);
	cout << "Pushing new cell" << endl;
	
	cells.push_back(mb);
	
	myfile << 0 << " " << cells.size() << endl;
	myfile << 0 << " " << 0 << " " << init_z << " "
				<< nxi << " " << nyi << " " << nzi << " "
							<< init_l << " " << cells[0]->get_ai() << " " << 1 << " " << 0 << endl;
	
	cout << "Calling grow" << endl;
	
	simple_grow(T);
	
	
	cout << "Done" << endl;
	
	return 0;
}



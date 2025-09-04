
#ifndef FRICP_H
#define FRICP_H

#include "Registration.h"
// #include "ICP.h"
//#include <AndersonAcceleration.h>
#include <eigen/unsupported/Eigen/MatrixFunctions>
#include "median.h"
#include <limits>
#define SAME_THRESHOLD 1e-6
#include <type_traits>
#include "Types.h"
#include <cassert>
#include <algorithm>
#include <vector>
#include <omp.h>
#include <fstream>
#include <eigen/Eigen/Dense>
#include <nanoflann.hpp>
#include <time.h>
#include <iostream>
#include <string>
#include "nanoflann.h"
#define TUPLE_SCALE	  0.95
#define TUPLE_MAX_CNT 1000
#ifdef USE_FLOAT_SCALAR
typedef float Scalar;
#else
typedef double Scalar;
#endif

#ifdef EIGEN_DONT_ALIGN
#define EIGEN_ALIGNMENT Eigen::DontAlign
#else
#define EIGEN_ALIGNMENT Eigen::AutoAlign
#endif



class AndersonAcceleration
{
public:
	AndersonAcceleration()
		:m_(-1), dim_(-1), iter_(-1), col_idx_(-1) {}

	void replace(const Scalar *u)
	{
		current_u_ = Eigen::Map<const VectorX>(u, dim_);
	}

	const VectorX& compute(const Scalar* g)
	{
		assert(iter_ >= 0);

		Eigen::Map<const VectorX> G(g, dim_);
		current_F_ = G - current_u_;

		if (iter_ == 0)
		{
			prev_dF_.col(0) = -current_F_;
			prev_dG_.col(0) = -G;
			current_u_ = G;
		}
		else
		{
			prev_dF_.col(col_idx_) += current_F_;
			prev_dG_.col(col_idx_) += G;

			Scalar eps = 1e-14;
			Scalar scale = std::max(eps, prev_dF_.col(col_idx_).norm());
			dF_scale_(col_idx_) = scale;
			prev_dF_.col(col_idx_) /= scale;

			int m_k = std::min(m_, iter_);


			if (m_k == 1)
			{
				theta_(0) = 0;
				Scalar dF_sqrnorm = prev_dF_.col(col_idx_).squaredNorm();
				M_(0, 0) = dF_sqrnorm;
				Scalar dF_norm = std::sqrt(dF_sqrnorm);

                if (dF_norm > eps) {
					theta_(0) = (prev_dF_.col(col_idx_) / dF_norm).dot(current_F_ / dF_norm);
				}
			}
			else
			{
				// Update the normal equation matrix, for the column and row corresponding to the new dF column
				VectorX new_inner_prod = (prev_dF_.col(col_idx_).transpose() * prev_dF_.block(0, 0, dim_, m_k)).transpose();
				M_.block(col_idx_, 0, 1, m_k) = new_inner_prod.transpose();
				M_.block(0, col_idx_, m_k, 1) = new_inner_prod;

				// Solve normal equation
				cod_.compute(M_.block(0, 0, m_k, m_k));
				theta_.head(m_k) = cod_.solve(prev_dF_.block(0, 0, dim_, m_k).transpose() * current_F_);
			}

			// Use rescaled theata to compute new u
			current_u_ = G - prev_dG_.block(0, 0, dim_, m_k) * ((theta_.head(m_k).array() / dF_scale_.head(m_k).array()).matrix());
			col_idx_ = (col_idx_ + 1) % m_;
			prev_dF_.col(col_idx_) = -current_F_;
			prev_dG_.col(col_idx_) = -G;
		}

		iter_++;
		return current_u_;
	};
    void reset(const Scalar *u)
    {
        iter_ = 0;
        col_idx_ = 0;
        current_u_ = Eigen::Map<const VectorX>(u, dim_);
    }

	// m: number of previous iterations used
	// d: dimension of variables
	// u0: initial variable values
	void init(int m, int d, const Scalar* u0)
	{
		assert(m > 0);
		m_ = m;
		dim_ = d;
		current_u_.resize(d);
		current_F_.resize(d);
		prev_dG_.resize(d, m);
		prev_dF_.resize(d, m);
		M_.resize(m, m);
		theta_.resize(m);
		dF_scale_.resize(m);
		current_u_ = Eigen::Map<const VectorX>(u0, d);
		iter_ = 0;
		col_idx_ = 0;
	}

private:
	VectorX current_u_;
	VectorX current_F_;
	MatrixXX prev_dG_;
	MatrixXX prev_dF_;
	MatrixXX M_;		// Normal equations matrix for the computing theta
	VectorX	theta_;	// theta value computed from normal equations
	VectorX dF_scale_;		// The scaling factor for each column of prev_dF
	Eigen::CompleteOrthogonalDecomposition<MatrixXX> cod_;

	int m_;		// Number of previous iterates used for Andreson Acceleration
	int dim_;	// Dimension of variables
	int iter_;	// Iteration count since initialization
	int col_idx_;	// Index for history matrix column to store the next value
	int m_k_;

	Eigen::Matrix4d current_T_;
	Eigen::Matrix4d current_F_T_;

	MatrixXX T_prev_dF_;
	MatrixXX T_prev_dG_;
};


namespace ICP{
    enum Function {
        PNORM,
        TUKEY,
        FAIR,
        LOGISTIC,
        TRIMMED,
        WELSCH,
        AUTOWELSCH,
        NONE
    };
    class Parameters {
    public:
        Parameters() : 
            f(NONE),
            p(0.1),
            max_icp(100),
            max_outer(1),
            stop(1e-5),
            use_AA(false),
            print_energy(false),
            print_output(false),
            anderson_m(5),
            beta_(1.0),
            error_overflow_threshold_(0.05),
            has_groundtruth(false),
            convergence_energy(0.0),
            convergence_iter(0),
            convergence_gt_mse(0.0),
            nu_begin_k(3),
            nu_end_k(1.0 / (3.0 * sqrt(3.0))),
            use_init(false),
            nu_alpha(1.0 / 2)
        {
            // 矩阵成员在构造函数体内赋值，避免 most vexing parse
            gt_trans = Eigen::Matrix4d::Identity();
            init_trans = Eigen::MatrixXd(); // 或指定大小
            res_trans = Eigen::MatrixXd();
        }

        /// Parameters
        Function f;     /// robust function type
        double p;       /// paramter of the robust function/// para k
        int max_icp;    /// max ICP iteration
        int max_outer;  /// max outer iteration
        double stop;    /// stopping criteria
        bool use_AA;  /// whether using anderson acceleration
        std::string out_path;
        bool print_energy;///whether print energy
        bool print_output; ///whether write result to txt
        int anderson_m;
        double beta_;
        double error_overflow_threshold_;
        MatrixXX init_trans;
        MatrixXX gt_trans;
        bool has_groundtruth;
        double convergence_energy;
        int convergence_iter;
        double convergence_gt_mse;
        MatrixXX res_trans;
        double nu_begin_k;
        double nu_end_k;
        bool use_init;
        double nu_alpha;
    };
    /// Weight functions
    /// @param Residuals
    /// @param Parameter
    struct sort_pred {
        bool operator()(const std::pair<int, double> &left,
            const std::pair<int, double> &right) {
            return left.second < right.second;
        }
    };
    
inline void uniform_weight(Eigen::VectorXd& r) {
        r = Eigen::VectorXd::Ones(r.rows());
    }
    /// @param Residuals
    /// @param Parameter
inline void pnorm_weight(Eigen::VectorXd& r, double p, double reg=1e-8 ) {
        for (int i = 0; i<r.rows(); ++i) {
            r(i) = p / (std::pow(r(i), 2 - p) + reg);
        }
    }
    /// @param Residuals
    /// @param Parameter
    inline void tukey_weight(Eigen::VectorXd& r, double p) {
        for (int i = 0; i<r.rows(); ++i) {
            if (r(i) > p) r(i) = 0.0;
            else r(i) = std::pow((1.0 - std::pow(r(i) / p, 2.0)), 2.0);
        }
    }
    /// @param Residuals
    /// @param Parameter
  inline  void fair_weight(Eigen::VectorXd& r, double p) {
        for (int i = 0; i<r.rows(); ++i) {
            r(i) = 1.0 / (1.0 + r(i) / p);
        }
    }
    /// @param Residuals
    /// @param Parameter
  inline  void logistic_weight(Eigen::VectorXd& r, double p) {
        for (int i = 0; i<r.rows(); ++i) {
            r(i) = (p / r(i))*std::tanh(r(i) / p);
        }
    }
    
    /// @param Residuals
    /// @param Parameter
  inline  void trimmed_weight(Eigen::VectorXd& r, double p) {
        std::vector<std::pair<int, double> > sortedDist(r.rows());
        for (int i = 0; i<r.rows(); ++i) {
            sortedDist[i] = std::pair<int, double>(i, r(i));
        }
        std::sort(sortedDist.begin(), sortedDist.end(), sort_pred());
        r.setZero();
        int nbV = r.rows()*p;
        for (int i = 0; i<nbV; ++i) {
            r(sortedDist[i].first) = 1.0;
        }
    }
    /// @param Residuals
   inline void welsch_weight(Eigen::VectorXd& r, double p) {
        for (int i = 0; i<r.rows(); ++i) {
            r(i) = std::exp(-r(i)*r(i)/(2*p*p));
        }
    }

    /// @param Residuals
    /// @param Parameter
  inline  void autowelsch_weight(Eigen::VectorXd& r, double p) {
        double median;
        igl::median(r, median);
        welsch_weight(r, p*median/(std::sqrt(2)*2.3));
        //welsch_weight(r,p);
    }

    /// Energy functions
    /// @param Residuals
    /// @param Parameter
  inline  double uniform_energy(Eigen::VectorXd& r) {
        double energy = 0;
        for (int i = 0; i<r.rows(); ++i) {
            energy += r(i)*r(i);
        }
        return energy;
    }
    /// @param Residuals
    /// @param Parameter
   inline double pnorm_energy(Eigen::VectorXd& r, double p, double reg=1e-8) {
        double energy = 0;
        for (int i = 0; i<r.rows(); ++i) {
            energy += (r(i)*r(i))*p / (std::pow(r(i), 2 - p) + reg);
        }
        return energy;
    }
    /// @param Residuals
    /// @param Parameter
    inline double tukey_energy(Eigen::VectorXd& r, double p) {
        double energy = 0;
        double w;
        for (int i = 0; i<r.rows(); ++i) {
            if (r(i) > p) w = 0.0;
            else w = std::pow((1.0 - std::pow(r(i) / p, 2.0)), 2.0);

            energy += (r(i)*r(i))*w;
        }
        return energy;
    }
    /// @param Residuals
    /// @param Parameter
   inline double fair_energy(Eigen::VectorXd& r, double p) {
        double energy = 0;
        for (int i = 0; i<r.rows(); ++i) {
            energy += (r(i)*r(i))*1.0 / (1.0 + r(i) / p);
        }
        return energy;
    }
    /// @param Residuals
    /// @param Parameter
   inline double logistic_energy(Eigen::VectorXd& r, double p) {
        double energy = 0;
        for (int i = 0; i<r.rows(); ++i) {
            energy += (r(i)*r(i))*(p / r(i))*std::tanh(r(i) / p);
        }
        return energy;
    }
    /// @param Residuals
    /// @param Parameter
   inline double trimmed_energy(Eigen::VectorXd& r, double p) {
        std::vector<std::pair<int, double> > sortedDist(r.rows());
        for (int i = 0; i<r.rows(); ++i) {
            sortedDist[i] = std::pair<int, double>(i, r(i));
        }
        std::sort(sortedDist.begin(), sortedDist.end(), sort_pred());
        Eigen::VectorXd t = r;
        t.setZero();
        double energy = 0;
        int nbV = r.rows()*p;
        for (int i = 0; i<nbV; ++i) {
            energy += r(i)*r(i);
        }
        return energy;
    }

    /// @param Residuals
    /// @param Parameter
  inline  double welsch_energy(Eigen::VectorXd& r, double p) {
        double energy = 0;
        for (int i = 0; i<r.rows(); ++i) {
            energy += 1.0 - std::exp(-r(i)*r(i)/(2*p*p));
        }
        return energy;
    }
    /// @param Residuals
    /// @param Parameter
   inline double autowelsch_energy(Eigen::VectorXd& r, double p) {
        double energy = 0;
        energy = welsch_energy(r, 0.5);
        return energy;
    }
    /// @param Function type
    /// @param Residuals
    /// @param Parameter
   inline void robust_weight(Function f, Eigen::VectorXd& r, double p) {
        switch (f) {
        case PNORM: pnorm_weight(r, p); break;
        case TUKEY: tukey_weight(r, p); break;
        case FAIR: fair_weight(r, p); break;
        case LOGISTIC: logistic_weight(r, p); break;
        case TRIMMED: trimmed_weight(r, p); break;
        case WELSCH: welsch_weight(r, p); break;
        case AUTOWELSCH: autowelsch_weight(r,p); break;
        case NONE: uniform_weight(r); break;
        default: uniform_weight(r); break;
        }
    }


    //Cacl energy
  inline  double get_energy(Function f, Eigen::VectorXd& r, double p) {
        double energy = 0;
        switch (f) {
            //case PNORM: pnorm_weight(r,p); break;
        case TUKEY: energy = tukey_energy(r, p); break;
        case FAIR: energy = fair_energy(r, p); break;
        case LOGISTIC: energy = logistic_energy(r, p); break;
        case TRIMMED: energy = trimmed_energy(r, p); break;
        case WELSCH: energy = welsch_energy(r, p); break;
        case AUTOWELSCH: energy = autowelsch_energy(r, p); break;
        case NONE: energy = uniform_energy(r); break;
        default: energy = uniform_energy(r); break;
        }
        return energy;
    }
    
    
}


template<class T>
typename std::enable_if<!std::numeric_limits<T>::is_integer, bool>::type
almost_equal(T x, T y, int ulp)
{
    // the machine epsilon has to be scaled to the magnitude of the values used
    // and multiplied by the desired precision in ULPs (units in the last place)
    return std::fabs(x-y) <= std::numeric_limits<T>::epsilon() * std::fabs(x+y) * ulp
            // unless the result is subnormal
            || std::fabs(x-y) < std::numeric_limits<T>::min();
}

template<int N>
class FRICP
{
public:
    typedef double Scalar;
    typedef Eigen::Matrix<Scalar, N, Eigen::Dynamic> MatrixNX;
    typedef Eigen::Matrix<Scalar, N, N> MatrixNN;
    typedef Eigen::Matrix<Scalar, N+1, N+1> AffineMatrixN;
    typedef Eigen::Transform<Scalar, N, Eigen::Affine> AffineNd;
    typedef Eigen::Matrix<Scalar, N, 1> VectorN;
    typedef nanoflann::KDTreeAdaptor<MatrixNX, N, nanoflann::metric_L2_Simple> KDtree;
    typedef Eigen::Matrix<Scalar, 6, 1> Vector6;
    double test_total_construct_time=.0;
    double test_total_solve_time=.0;
    int test_total_iters=0;
    
    FRICP(){};
    ~FRICP(){};

private:
    AffineMatrixN LogMatrix(const AffineMatrixN& T)
    {
        Eigen::RealSchur<AffineMatrixN> schur(T);
        AffineMatrixN U = schur.matrixU();
        AffineMatrixN R = schur.matrixT();
        std::vector<bool> selected(N, true);
        MatrixNN mat_B = MatrixNN::Zero(N, N);
        MatrixNN mat_V = MatrixNN::Identity(N, N);

        for (int i = 0; i < N; i++)
        {
            if (selected[i] && fabs(R(i, i) - 1)> SAME_THRESHOLD)
            {
                int pair_second = -1;
                for (int j = i + 1; j <N; j++)
                {
                    if (fabs(R(j, j) - R(i, i)) < SAME_THRESHOLD)
                    {
                        pair_second = j;
                        selected[j] = false;
                        break;
                    }
                }
                if (pair_second > 0)
                {
                    selected[i] = false;
                    R(i, i) = R(i, i) < -1 ? -1 : R(i, i);
                    double theta = acos(R(i, i));
                    if (R(i, pair_second) < 0)
                    {
                        theta = -theta;
                    }
                    mat_B(i, pair_second) += theta;
                    mat_B(pair_second, i) += -theta;
                    mat_V(i, pair_second) += -theta / 2;
                    mat_V(pair_second, i) += theta / 2;
                    double coeff = 1 - (theta * R(i, pair_second)) / (2 * (1 - R(i, i)));
                    mat_V(i, i) += -coeff;
                    mat_V(pair_second, pair_second) += -coeff;
                }
            }
        }

        AffineMatrixN LogTrim = AffineMatrixN::Zero();
        LogTrim.block(0, 0, N, N) = mat_B;
        LogTrim.block(0, N, N, 1) = mat_V * R.block(0, N, N, 1);
        AffineMatrixN res = U * LogTrim * U.transpose();
        return res;
    }

    inline Vector6 RotToEuler(const AffineNd& T)
    {
        Vector6 res;
        res.head(3) = T.rotation().eulerAngles(0,1,2);
        res.tail(3) = T.translation();
        return res;
    }

    inline AffineMatrixN EulerToRot(const Vector6& v)
    {
        MatrixNN s (Eigen::AngleAxis<Scalar>(v(0), Vector3::UnitX())
                    * Eigen::AngleAxis<Scalar>(v(1), Vector3::UnitY())
                    * Eigen::AngleAxis<Scalar>(v(2), Vector3::UnitZ()));

        AffineMatrixN m = AffineMatrixN::Zero();
        m.block(0,0,3,3) = s;
        m(3,3) = 1;
        m.col(3).head(3) = v.tail(3);
        return m;
    }
    inline Vector6 LogToVec(const Eigen::Matrix4d& LogT)
    {
        Vector6 res;
        res[0] = -LogT(1, 2);
        res[1] = LogT(0, 2);
        res[2] = -LogT(0, 1);
        res[3] = LogT(0, 3);
        res[4] = LogT(1, 3);
        res[5] = LogT(2, 3);
        return res;
    }

    inline AffineMatrixN VecToLog(const Vector6& v)
    {
        AffineMatrixN m = AffineMatrixN::Zero();
        m << 0, -v[2], v[1], v[3],
                v[2], 0, -v[0], v[4],
                -v[1], v[0], 0, v[5],
                0, 0, 0, 0;
        return m;
    }

    double FindKnearestMed(const KDtree& kdtree,
                           const MatrixNX& X, int nk)
    {
        Eigen::VectorXd X_nearest(X.cols());
#pragma omp parallel for
        for(int i = 0; i<X.cols(); i++)
        {
            int* id = new int[nk];
            double *dist = new double[nk];
            kdtree.query(X.col(i).data(), nk, id, dist);
            Eigen::VectorXd k_dist = Eigen::Map<Eigen::VectorXd>(dist, nk);
            igl::median(k_dist.tail(nk-1), X_nearest[i]);
            delete[]id;
            delete[]dist;
        }
        double med;
        igl::median(X_nearest, med);
        return sqrt(med);
    }
    /// Find self normal edge median of point cloud
    double FindKnearestNormMed(const KDtree& kdtree, const Eigen::Matrix3Xd & X, int nk, const Eigen::Matrix3Xd & norm_x)
    {
        Eigen::VectorXd X_nearest(X.cols());
#pragma omp parallel for
        for(int i = 0; i<X.cols(); i++)
        {
            int* id = new int[nk];
            double *dist = new double[nk];
            kdtree.query(X.col(i).data(), nk, id, dist);
            Eigen::VectorXd k_dist = Eigen::Map<Eigen::VectorXd>(dist, nk);
            for(int s = 1; s<nk; s++)
            {
                k_dist[s] = std::abs((X.col(id[s]) - X.col(id[0])).dot(norm_x.col(id[0])));
            }
            igl::median(k_dist.tail(nk-1), X_nearest[i]);
            delete[]id;
            delete[]dist;
        }
        double med;
        igl::median(X_nearest, med);
        return med;
    }

    template <typename Derived1, typename Derived2, typename Derived3>
    AffineNd point_to_point(Eigen::MatrixBase<Derived1>& X,
                            Eigen::MatrixBase<Derived2>& Y,
                            const Eigen::MatrixBase<Derived3>& w) {
        int dim = X.rows();
        /// Normalize weight vector
        Eigen::VectorXd w_normalized = w / w.sum();
        /// De-mean
        Eigen::VectorXd X_mean(dim), Y_mean(dim);
        for (int i = 0; i<dim; ++i) {
            X_mean(i) = (X.row(i).array()*w_normalized.transpose().array()).sum();
            Y_mean(i) = (Y.row(i).array()*w_normalized.transpose().array()).sum();
        }
        X.colwise() -= X_mean;
        Y.colwise() -= Y_mean;

        /// Compute transformation
        AffineNd transformation;
        MatrixXX sigma = X * w_normalized.asDiagonal() * Y.transpose();
        Eigen::JacobiSVD<MatrixXX> svd(sigma, Eigen::ComputeFullU | Eigen::ComputeFullV);
        if (svd.matrixU().determinant()*svd.matrixV().determinant() < 0.0) {
            VectorN S = VectorN::Ones(dim); S(dim-1) = -1.0;
            transformation.linear() = svd.matrixV()*S.asDiagonal()*svd.matrixU().transpose();
        }
        else {
            transformation.linear() = svd.matrixV()*svd.matrixU().transpose();
        }
        transformation.translation() = Y_mean - transformation.linear()*X_mean;
        /// Re-apply mean
        X.colwise() += X_mean;
        Y.colwise() += Y_mean;
        /// Return transformation
        return transformation;
    }

    template <typename Derived1, typename Derived2, typename Derived3, typename Derived4, typename Derived5>
    Eigen::Affine3d point_to_plane(Eigen::MatrixBase<Derived1>& X,
                                   Eigen::MatrixBase<Derived2>& Y,
                                   const Eigen::MatrixBase<Derived3>& Norm,
                                   const Eigen::MatrixBase<Derived4>& w,
                                   const Eigen::MatrixBase<Derived5>& u) {
        typedef Eigen::Matrix<double, 6, 6> Matrix66;
        typedef Eigen::Matrix<double, 6, 1> Vector6;
        typedef Eigen::Block<Matrix66, 3, 3> Block33;
        /// Normalize weight vector
        Eigen::VectorXd w_normalized = w / w.sum();
        /// De-mean
        Eigen::Vector3d X_mean;
        for (int i = 0; i<3; ++i)
            X_mean(i) = (X.row(i).array()*w_normalized.transpose().array()).sum();
        X.colwise() -= X_mean;
        Y.colwise() -= X_mean;
        /// Prepare LHS and RHS
        Matrix66 LHS = Matrix66::Zero();
        Vector6 RHS = Vector6::Zero();
        Block33 TL = LHS.topLeftCorner<3, 3>();
        Block33 TR = LHS.topRightCorner<3, 3>();
        Block33 BR = LHS.bottomRightCorner<3, 3>();
        Eigen::MatrixXd C = Eigen::MatrixXd::Zero(3, X.cols());

#pragma omp parallel
        {
#pragma omp for
            for (int i = 0; i<X.cols(); i++) {
                C.col(i) = X.col(i).cross(Norm.col(i));
            }
#pragma omp sections nowait
            {
#pragma omp section
                for (int i = 0; i<X.cols(); i++) TL.selfadjointView<Eigen::Upper>().rankUpdate(C.col(i), w(i));
#pragma omp section
                for (int i = 0; i<X.cols(); i++) TR += (C.col(i)*Norm.col(i).transpose())*w(i);
#pragma omp section
                for (int i = 0; i<X.cols(); i++) BR.selfadjointView<Eigen::Upper>().rankUpdate(Norm.col(i), w(i));
#pragma omp section
                for (int i = 0; i<C.cols(); i++) {
                    double dist_to_plane = -((X.col(i) - Y.col(i)).dot(Norm.col(i)) - u(i))*w(i);
                    RHS.head<3>() += C.col(i)*dist_to_plane;
                    RHS.tail<3>() += Norm.col(i)*dist_to_plane;
                }
            }
        }
        LHS = LHS.selfadjointView<Eigen::Upper>();
        /// Compute transformation
        Eigen::Affine3d transformation;
        Eigen::LDLT<Matrix66> ldlt(LHS);
        RHS = ldlt.solve(RHS);
        transformation = Eigen::AngleAxisd(RHS(0), Eigen::Vector3d::UnitX()) *
                Eigen::AngleAxisd(RHS(1), Eigen::Vector3d::UnitY()) *
                Eigen::AngleAxisd(RHS(2), Eigen::Vector3d::UnitZ());
        transformation.translation() = RHS.tail<3>();

        /// Apply transformation
        /// Re-apply mean
        X.colwise() += X_mean;
        Y.colwise() += X_mean;
        transformation.translation() += X_mean - transformation.linear()*X_mean;
        /// Return transformation
        return transformation;
    }

    template <typename Derived1, typename Derived2, typename Derived3, typename Derived4>
    double point_to_plane_gaussnewton(const Eigen::MatrixBase<Derived1>& X,
                               const Eigen::MatrixBase<Derived2>& Y,
                               const Eigen::MatrixBase<Derived3>& norm_y,
                               const Eigen::MatrixBase<Derived4>& w,
                               Matrix44 Tk,  Vector6& dir) {
        typedef Eigen::Matrix<double, 6, 6> Matrix66;
        typedef Eigen::Matrix<double, 12, 6> Matrix126;
        typedef Eigen::Matrix<double, 9, 3> Matrix93;
        typedef Eigen::Block<Matrix126, 9, 3> Block93;
        typedef Eigen::Block<Matrix126, 3, 3> Block33;
        typedef Eigen::Matrix<double, 12, 1> Vector12;
        typedef Eigen::Matrix<double, 9, 1> Vector9;
        typedef Eigen::Matrix<double, 4, 2> Matrix42;
        /// Normalize weight vector
        Eigen::VectorXd w_normalized = w / w.sum();
        /// Prepare LHS and RHS
        Matrix66 LHS = Matrix66::Zero();
        Vector6 RHS = Vector6::Zero();

        Vector6 log_T = LogToVec(LogMatrix(Tk));
        Matrix33 B = VecToLog(log_T).block(0, 0, 3, 3);
        double a = log_T[0];
        double b = log_T[1];
        double c = log_T[2];
        Matrix33 R = Tk.block(0, 0, 3, 3);
        Vector3 t = Tk.block(0, 3, 3, 1);
        Vector3 u = log_T.tail(3);

        Matrix93 dbdw = Matrix93::Zero();
        dbdw(1, 2) = dbdw(5, 0) = dbdw(6, 1) = -1;
        dbdw(2, 1) = dbdw(3, 2) = dbdw(7, 0) = 1;
        Matrix93 db2dw = Matrix93::Zero();
        db2dw(3, 1) = db2dw(4, 0) = db2dw(6, 2) = db2dw(8, 0) = a;
        db2dw(0, 1) = db2dw(1, 0) = db2dw(7, 2) = db2dw(8, 1) = b;
        db2dw(0, 2) = db2dw(2, 0) = db2dw(4, 2) = db2dw(5, 1) = c;
        db2dw(1, 1) = db2dw(2, 2) = -2 * a;
        db2dw(3, 0) = db2dw(5, 2) = -2 * b;
        db2dw(6, 0) = db2dw(7, 1) = -2 * c;
        double theta = std::sqrt(a*a + b*b + c*c);
        double st = sin(theta), ct = cos(theta);

        Matrix42 coeff = Matrix42::Zero();
        if (theta>SAME_THRESHOLD)
        {
            coeff << st / theta, (1 - ct) / (theta*theta),
                    (theta*ct - st) / (theta*theta*theta), (theta*st - 2 * (1 - ct)) / pow(theta, 4),
                    (1 - ct) / (theta*theta), (theta - st) / pow(theta, 3),
                    (theta*st - 2 * (1 - ct)) / pow(theta, 4), (theta*(1 - ct) - 3 * (theta - st)) / pow(theta, 5);
        }
        else
            coeff(0, 0) = 1;

        Matrix93 tempB3;
        tempB3.block<3, 3>(0, 0) = a*B;
        tempB3.block<3, 3>(3, 0) = b*B;
        tempB3.block<3, 3>(6, 0) = c*B;
        Matrix33 B2 = B*B;
        Matrix93 temp2B3;
        temp2B3.block<3, 3>(0, 0) = a*B2;
        temp2B3.block<3, 3>(3, 0) = b*B2;
        temp2B3.block<3, 3>(6, 0) = c*B2;
        Matrix93 dRdw = coeff(0, 0)*dbdw + coeff(1, 0)*tempB3
                + coeff(2, 0)*db2dw + coeff(3, 0)*temp2B3;
        Vector9 dtdw = coeff(0, 1) * dbdw*u + coeff(1, 1) * tempB3*u
                + coeff(2, 1) * db2dw*u + coeff(3, 1)*temp2B3*u;
        Matrix33 dtdu = Matrix33::Identity() + coeff(2, 0)*B + coeff(2, 1) * B2;

        Eigen::VectorXd rk(X.cols());
        Eigen::MatrixXd Jk(X.cols(), 6);
#pragma omp for
        for (int i = 0; i < X.cols(); i++)
        {
            Vector3 xi = X.col(i);
            Vector3 yi = Y.col(i);
            Vector3 ni = norm_y.col(i);
            double wi = sqrt(w_normalized[i]);

            Matrix33 dedR = wi*ni * xi.transpose();
            Vector3 dedt = wi*ni;

            Vector6 dedx;
            dedx(0) = (dedR.cwiseProduct(dRdw.block(0, 0, 3, 3))).sum()
                    + dedt.dot(dtdw.head<3>());
            dedx(1) = (dedR.cwiseProduct(dRdw.block(3, 0, 3, 3))).sum()
                    + dedt.dot(dtdw.segment<3>(3));
            dedx(2) = (dedR.cwiseProduct(dRdw.block(6, 0, 3, 3))).sum()
                    + dedt.dot(dtdw.tail<3>());
            dedx(3) = dedt.dot(dtdu.col(0));
            dedx(4) = dedt.dot(dtdu.col(1));
            dedx(5) = dedt.dot(dtdu.col(2));

            Jk.row(i) = dedx.transpose();
            rk[i] = wi * ni.dot(R*xi-yi+t);
        }
        LHS = Jk.transpose() * Jk;
        RHS = -Jk.transpose() * rk;
        Eigen::CompleteOrthogonalDecomposition<Matrix66> cod_(LHS);
        dir = cod_.solve(RHS);
        double gTd = -RHS.dot(dir);
        return gTd;
    }


public:
     void point_to_point(MatrixNX& X, MatrixNX& Y, VectorN& source_mean,
                        VectorN& target_mean, ICP::Parameters& par){
        /// Build kd-tree
        KDtree kdtree(Y);
        /// Buffers
        MatrixNX Q = MatrixNX::Zero(N, X.cols());
        VectorX W = VectorX::Zero(X.cols());
        AffineNd T;
        if (par.use_init) T.matrix() = par.init_trans;
        else T = AffineNd::Identity();
        MatrixXX To1 = T.matrix();
        MatrixXX To2 = T.matrix();
        int nPoints = X.cols();

        //Anderson Acc para
        AndersonAcceleration accelerator_;
        AffineNd SVD_T = T;
        double energy = .0, last_energy = std::numeric_limits<double>::max();

        //ground truth point clouds
        MatrixNX X_gt = X;
        if(par.has_groundtruth)
        {
            VectorN temp_trans = par.gt_trans.col(N).head(N);
            X_gt.colwise() += source_mean;
            X_gt = par.gt_trans.block(0, 0, N, N) * X_gt;
            X_gt.colwise() += temp_trans - target_mean;
        }

        //output para
        std::string file_out = par.out_path;
        std::vector<double> times, energys, gt_mses;
        double begin_time, end_time, run_time;
        double gt_mse = 0.0;

        // dynamic welsch paras
        double nu1 = 1, nu2 = 1;
        double begin_init = omp_get_wtime();

        //Find initial closest point
#pragma omp parallel for
        for (int i = 0; i<nPoints; ++i) {
            VectorN cur_p = T * X.col(i);
            Q.col(i) = Y.col(kdtree.closest(cur_p.data()));
            W[i] = (cur_p - Q.col(i)).norm();
        }
        if(par.f == ICP::WELSCH)
        {
            //dynamic welsch, calc k-nearest points with itself;
            nu2 = par.nu_end_k * FindKnearestMed(kdtree, Y, 7);
            double med1;
            igl::median(W, med1);
            nu1 = par.nu_begin_k * med1;
            nu1 = nu1>nu2? nu1:nu2;
        }
        double end_init = omp_get_wtime();
        double init_time = end_init - begin_init;

        //AA init
        accelerator_.init(par.anderson_m, (N + 1) * (N + 1), LogMatrix(T.matrix()).data());

        begin_time = omp_get_wtime();
        bool stop1 = false;
        while(!stop1)
        {
            /// run ICP
            int icp = 0;
            for (; icp<par.max_icp; ++icp)
            {
                bool accept_aa = false;
                energy = get_energy(par.f, W, nu1);
                if (par.use_AA)
                {
                    if (energy < last_energy) {
                        last_energy = energy;
                        accept_aa = true;
                    }
                    else{
                        accelerator_.replace(LogMatrix(SVD_T.matrix()).data());
                        //Re-find the closest point
#pragma omp parallel for
                        for (int i = 0; i<nPoints; ++i) {
                            VectorN cur_p = SVD_T * X.col(i);
                            Q.col(i) = Y.col(kdtree.closest(cur_p.data()));
                            W[i] = (cur_p - Q.col(i)).norm();
                        }
                        last_energy = get_energy(par.f, W, nu1);
                    }
                }
                else
                    last_energy = energy;

                end_time = omp_get_wtime();
                run_time = end_time - begin_time;
                if(par.has_groundtruth)
                {
                    gt_mse = (T*X - X_gt).squaredNorm()/nPoints;
                }

                // save results
                energys.push_back(last_energy);
                times.push_back(run_time);
                gt_mses.push_back(gt_mse);

                if (par.print_energy)
                    std::cout << "icp iter = " << icp << ", Energy = " << last_energy
                             << ", time = " << run_time << std::endl;

                robust_weight(par.f, W, nu1);
                // Rotation and translation update
                T = point_to_point(X, Q, W);

                //Anderson Acc
                SVD_T = T;
                if (par.use_AA)
                {
                    AffineMatrixN Trans = (Eigen::Map<const AffineMatrixN>(accelerator_.compute(LogMatrix(T.matrix()).data()).data(), N+1, N+1)).exp();
                    T.linear() = Trans.block(0,0,N,N);
                    T.translation() = Trans.block(0,N,N,1);
                }

                // Find closest point
#pragma omp parallel for
                for (int i = 0; i<nPoints; ++i) {
                    VectorN cur_p = T * X.col(i) ;
                    Q.col(i) = Y.col(kdtree.closest(cur_p.data()));
                    W[i] = (cur_p - Q.col(i)).norm();
                }
                /// Stopping criteria
                double stop2 = (T.matrix() - To2).norm();
                To2 = T.matrix();
                if(stop2 < par.stop)
                {
                    break;
                }
            }
            if(par.f!= ICP::WELSCH)
                stop1 = true;
            else
            {
                stop1 = fabs(nu1 - nu2)<SAME_THRESHOLD? true: false;
                nu1 = nu1*par.nu_alpha > nu2? nu1*par.nu_alpha : nu2;
                if(par.use_AA)
                {
                    accelerator_.reset(LogMatrix(T.matrix()).data());
                    last_energy = std::numeric_limits<double>::max();
                }
            }
        }

        ///calc convergence energy
        last_energy = get_energy(par.f, W, nu1);
        X = T * X;
        gt_mse = (X-X_gt).squaredNorm()/nPoints;
        T.translation() += - T.rotation() * source_mean + target_mean;
        X.colwise() += target_mean;

        ///save convergence result
        par.convergence_energy = last_energy;
        par.convergence_gt_mse = gt_mse;
        par.res_trans = T.matrix();

        ///output
        if (par.print_output)
        {
            std::ofstream out_res(par.out_path);
            if (!out_res.is_open())
            {
                std::cout << "Can't open out file " << par.out_path << std::endl;
            }

            //output time and energy
            out_res.precision(16);
            for (int i = 0; i<times.size(); i++)
            {
                out_res << times[i] << " "<< energys[i] << " " << gt_mses[i] << std::endl;
            }
            out_res.close();
            std::cout << " write res to " << par.out_path << std::endl;
        }
    }


    /// Reweighted ICP with point to plane
    /// @param Source (one 3D point per column)
    /// @param Target (one 3D point per column)
    /// @param Target normals (one 3D normal per column)
    /// @param Parameters
    //    template <typename Derived1, typename Derived2, typename Derived3>
    void point_to_plane(Eigen::Matrix3Xd& X,
                        Eigen::Matrix3Xd& Y, Eigen::Matrix3Xd& norm_x, Eigen::Matrix3Xd& norm_y,
                        Eigen::Vector3d& source_mean, Eigen::Vector3d& target_mean,
                        ICP::Parameters &par) {
        /// Build kd-tree
        KDtree kdtree(Y);
        /// Buffers
        Eigen::Matrix3Xd Qp = Eigen::Matrix3Xd::Zero(3, X.cols());
        Eigen::Matrix3Xd Qn = Eigen::Matrix3Xd::Zero(3, X.cols());
        Eigen::VectorXd W = Eigen::VectorXd::Zero(X.cols());
        Eigen::Matrix3Xd ori_X = X;
        AffineNd T;
        if (par.use_init) T.matrix() = par.init_trans;
        else T = AffineNd::Identity();
        AffineMatrixN To1 = T.matrix();
        X = T*X;

        Eigen::Matrix3Xd X_gt = X;
        if(par.has_groundtruth)
        {
            Eigen::Vector3d temp_trans = par.gt_trans.block(0, 3, 3, 1);
            X_gt = ori_X;
            X_gt.colwise() += source_mean;
            X_gt = par.gt_trans.block(0, 0, 3, 3) * X_gt;
            X_gt.colwise() += temp_trans - target_mean;
        }

        std::vector<double> times, energys, gt_mses;
        double begin_time, end_time, run_time;
        double gt_mse = 0.0;

        ///dynamic welsch, calc k-nearest points with itself;
        double begin_init = omp_get_wtime();

        //Anderson Acc para
        AndersonAcceleration accelerator_;
        AffineNd LG_T = T;
        double energy = 0.0, prev_res = std::numeric_limits<double>::max(), res = 0.0;


        // Find closest point
#pragma omp parallel for
        for (int i = 0; i<X.cols(); ++i) {
            int id = kdtree.closest(X.col(i).data());
            Qp.col(i) = Y.col(id);
            Qn.col(i) = norm_y.col(id);
            W[i] = std::abs(Qn.col(i).transpose() * (X.col(i) - Qp.col(i)));
        }
        double end_init = omp_get_wtime();
        double init_time = end_init - begin_init;

        begin_time = omp_get_wtime();
        int total_iter = 0;
        double test_total_time = 0.0;
        bool stop1 = false;
        while(!stop1)
        {
            /// ICP
            for(int icp=0; icp<par.max_icp; ++icp) {
                total_iter++;

                bool accept_aa = false;
                energy = get_energy(par.f, W, par.p);
                end_time = omp_get_wtime();
                run_time = end_time - begin_time;
                energys.push_back(energy);
                times.push_back(run_time);
                Eigen::VectorXd test_w = (X-Qp).colwise().norm();
                if(par.has_groundtruth)
                {
                    gt_mse = (X - X_gt).squaredNorm()/X.cols();
                }
                gt_mses.push_back(gt_mse);

                /// Compute weights
                robust_weight(par.f, W, par.p);
                /// Rotation and translation update
                T = point_to_plane(X, Qp, Qn, W, Eigen::VectorXd::Zero(X.cols()))*T;
                /// Find closest point
#pragma omp parallel for
                for(int i=0; i<X.cols(); i++) {
                    X.col(i) = T * ori_X.col(i);
                    int id = kdtree.closest(X.col(i).data());
                    Qp.col(i) = Y.col(id);
                    Qn.col(i) = norm_y.col(id);
                    W[i] = std::abs(Qn.col(i).transpose() * (X.col(i) - Qp.col(i)));
                }

                if(par.print_energy)
                    std::cout << "icp iter = " << total_iter << ", gt_mse = " << gt_mse
                              << ", energy = " << energy << std::endl;

                /// Stopping criteria
                double stop2 = (T.matrix() - To1).norm();
                To1 = T.matrix();
                if(stop2 < par.stop) break;
            }
            stop1 = true;
        }

        par.res_trans = T.matrix();

        ///calc convergence energy
        W = (Qn.array()*(X - Qp).array()).colwise().sum().abs().transpose();
        energy = get_energy(par.f, W, par.p);
        gt_mse = (X - X_gt).squaredNorm() / X.cols();
        T.translation().noalias() += -T.rotation()*source_mean + target_mean;
        X.colwise() += target_mean;
        norm_x = T.rotation()*norm_x;

        ///save convergence result
        par.convergence_energy = energy;
        par.convergence_gt_mse = gt_mse;
        par.res_trans = T.matrix();

        ///output
        if (par.print_output)
        {
            std::ofstream out_res(par.out_path);
            if (!out_res.is_open())
            {
                std::cout << "Can't open out file " << par.out_path << std::endl;
            }

            ///output time and energy
            out_res.precision(16);
            for (int i = 0; i<total_iter; i++)
            {
                out_res << times[i] << " "<< energys[i] << " " << gt_mses[i] << std::endl;
            }
            out_res.close();
            std::cout << " write res to " << par.out_path << std::endl;
        }
    }



    /// Reweighted ICP with point to plane
    /// @param Source (one 3D point per column)
    /// @param Target (one 3D point per column)
    /// @param Target normals (one 3D normal per column)
    /// @param Parameters
    //    template <typename Derived1, typename Derived2, typename Derived3>
    void point_to_plane_GN(Eigen::Matrix3Xd& X,
                           Eigen::Matrix3Xd& Y, Eigen::Matrix3Xd& norm_x, Eigen::Matrix3Xd& norm_y,
                           Eigen::Vector3d& source_mean, Eigen::Vector3d& target_mean,
                           ICP::Parameters &par) {
        /// Build kd-tree
        KDtree kdtree(Y);
        /// Buffers
        Eigen::Matrix3Xd Qp = Eigen::Matrix3Xd::Zero(3, X.cols());
        Eigen::Matrix3Xd Qn = Eigen::Matrix3Xd::Zero(3, X.cols());
        Eigen::VectorXd W = Eigen::VectorXd::Zero(X.cols());
        Eigen::Matrix3Xd ori_X = X;
        AffineNd T;
        if (par.use_init) T.matrix() = par.init_trans;
        else T = AffineNd::Identity();
        AffineMatrixN To1 = T.matrix();
        X = T*X;

        Eigen::Matrix3Xd X_gt = X;
        if(par.has_groundtruth)
        {
            Eigen::Vector3d temp_trans = par.gt_trans.block(0, 3, 3, 1);
            X_gt = ori_X;
            X_gt.colwise() += source_mean;
            X_gt = par.gt_trans.block(0, 0, 3, 3) * X_gt;
            X_gt.colwise() += temp_trans - target_mean;
        }

        std::vector<double> times, energys, gt_mses;
        double begin_time, end_time, run_time;
        double gt_mse;

        ///dynamic welsch, calc k-nearest points with itself;
        double nu1 = 1, nu2 = 1;
        double begin_init = omp_get_wtime();

        //Anderson Acc para
        AndersonAcceleration accelerator_;
        Vector6 LG_T;
        Vector6 Dir;
        //add time test
        double energy = 0.0, prev_energy = std::numeric_limits<double>::max();
        if(par.use_AA)
        {
            Eigen::Matrix4d log_T = LogMatrix(T.matrix());
            LG_T = LogToVec(log_T);
            accelerator_.init(par.anderson_m, 6, LG_T.data());
        }

        // Find closest point
#pragma omp parallel for
        for (int i = 0; i<X.cols(); ++i) {
            int id = kdtree.closest(X.col(i).data());
            Qp.col(i) = Y.col(id);
            Qn.col(i) = norm_y.col(id);
            W[i] = std::abs(Qn.col(i).transpose() * (X.col(i) - Qp.col(i)));
        }

        if(par.f == ICP::WELSCH)
        {
            double med1;
            igl::median(W, med1);
            nu1 =par.nu_begin_k * med1;
            nu2 = par.nu_end_k * FindKnearestNormMed(kdtree, Y, 7, norm_y);
            nu1 = nu1>nu2? nu1:nu2;
        }
        double end_init = omp_get_wtime();
        double init_time = end_init - begin_init;

        begin_time = omp_get_wtime();
        int total_iter = 0;
        double test_total_time = 0.0;
        bool stop1 = false;
        par.max_icp = 6;
        while(!stop1)
        {
            par.max_icp = std::min(par.max_icp+1, 10);
            /// ICP
            for(int icp=0; icp<par.max_icp; ++icp) {
                total_iter++;

                int n_linsearch = 0;
                energy = get_energy(par.f, W, nu1);
                if(par.use_AA)
                {
                    if(energy < prev_energy)
                    {
                        prev_energy = energy;
                    }
                    else
                    {
                        // line search
                        double alpha = 0.0;
                        Vector6 new_t = LG_T;
                        Eigen::VectorXd lowest_W = W;
                        Eigen::Matrix3Xd lowest_Qp = Qp;
                        Eigen::Matrix3Xd lowest_Qn = Qn;
                        Eigen::Affine3d lowest_T = T;
                        n_linsearch++;
                        alpha = 1;
                        new_t = LG_T + alpha * Dir;
                        T.matrix() = VecToLog(new_t).exp();
                        /// Find closest point
#pragma omp parallel for
                        for(int i=0; i<X.cols(); i++) {
                            X.col(i) = T * ori_X.col(i);
                            int id = kdtree.closest(X.col(i).data());
                            Qp.col(i) = Y.col(id);
                            Qn.col(i) = norm_y.col(id);
                            W[i] = std::abs(Qn.col(i).transpose() * (X.col(i) - Qp.col(i)));
                        }
                        double test_energy = get_energy(par.f, W, nu1);
                       if(test_energy < energy)
                        {
                            accelerator_.reset(new_t.data());
                            energy = test_energy;
                        }
                        else
                        {
                            Qp = lowest_Qp;
                            Qn = lowest_Qn;
                            W = lowest_W;
                            T = lowest_T;
                        }
                        prev_energy = energy;
                    }
                }
                else
                {
                    prev_energy = energy;
                }

                end_time = omp_get_wtime();
                run_time = end_time - begin_time;
                energys.push_back(prev_energy);
                times.push_back(run_time);
                if(par.has_groundtruth)
                {
                    gt_mse = (X - X_gt).squaredNorm()/X.cols();
                }
                gt_mses.push_back(gt_mse);

                /// Compute weights
                robust_weight(par.f, W, nu1);
                /// Rotation and translation update
                point_to_plane_gaussnewton(ori_X, Qp, Qn, W, T.matrix(), Dir);
                LG_T = LogToVec(LogMatrix(T.matrix()));
                LG_T += Dir;
                T.matrix() = VecToLog(LG_T).exp();

                // Anderson acc
                if(par.use_AA)
                {
                    Vector6 AA_t;
                    AA_t = accelerator_.compute(LG_T.data());
                    T.matrix() = VecToLog(AA_t).exp();
                }
                if(par.print_energy)
                    std::cout << "icp iter = " << total_iter << ", gt_mse = " << gt_mse
                              << ", nu1 = " << nu1 << ", acept_aa= " << n_linsearch
                              << ", energy = " << prev_energy << std::endl;

                /// Find closest point
#pragma omp parallel for
                for(int i=0; i<X.cols(); i++) {
                    X.col(i) = T * ori_X.col(i);
                    int id = kdtree.closest(X.col(i).data());
                    Qp.col(i) = Y.col(id);
                    Qn.col(i) = norm_y.col(id);
                    W[i] = std::abs(Qn.col(i).transpose() * (X.col(i) - Qp.col(i)));
                }

                /// Stopping criteria
                double stop2 = (T.matrix() - To1).norm();
                To1 = T.matrix();
                if(stop2 < par.stop) break;
            }

            if(par.f == ICP::WELSCH)
            {
                stop1 = fabs(nu1 - nu2)<SAME_THRESHOLD? true: false;
                nu1 = nu1*par.nu_alpha > nu2 ? nu1*par.nu_alpha : nu2;
                if(par.use_AA)
                {
                    accelerator_.reset(LogToVec(LogMatrix(T.matrix())).data());
                    prev_energy = std::numeric_limits<double>::max();
                }
            }
            else
                stop1 = true;
        }

        par.res_trans = T.matrix();

        ///calc convergence energy
        W = (Qn.array()*(X - Qp).array()).colwise().sum().abs().transpose();
        energy = get_energy(par.f, W, nu1);
        gt_mse = (X - X_gt).squaredNorm() / X.cols();
        T.translation().noalias() += -T.rotation()*source_mean + target_mean;
        X.colwise() += target_mean;
        norm_x = T.rotation()*norm_x;

        ///save convergence result
        par.convergence_energy = energy;
        par.convergence_gt_mse = gt_mse;
        par.res_trans = T.matrix();

        ///output
        if (par.print_output)
        {
            std::ofstream out_res(par.out_path);
            if (!out_res.is_open())
            {
                std::cout << "Can't open out file " << par.out_path << std::endl;
            }

            ///output time and energy
            out_res.precision(16);
            for (int i = 0; i<total_iter; i++)
            {
                out_res << times[i] << " "<< energys[i] << " " << gt_mses[i] << std::endl;
            }
            out_res.close();
            std::cout << " write res to " << par.out_path << std::endl;
        }
    }
};




class Rigid_FRICP: public REG
{

public:
    // 构造和析构
    Rigid_FRICP() = default;
    ~Rigid_FRICP() override = default;
    void Reg(const std::string& file_target,
                       const std::string& file_source,
                       const std::string& out_path) override;//配准函数
    
};

#endif
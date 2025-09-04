#ifndef QN_WELSCH_H_
#define QN_WELSCH_H_
#include "Registration.h"
#include <nanoflann.hpp>
#include "nanoflann.h"
#include <string>
#ifdef USE_PARDISO
#include <Eigen/PardisoSupport>
#endif
// #define DEBUG
#define USE_OPENMP

#ifdef USE_OPENMP
#include <omp.h>
#ifdef USE_MSVC
#define OMP_PARALLEL __pragma(omp parallel)
#define OMP_FOR __pragma(omp for)
#define OMP_SINGLE __pragma(omp single)
#define OMP_SECTIONS __pragma(omp sections)
#define OMP_SECTION __pragma(omp section)
#else
#define OMP_PARALLEL _Pragma("omp parallel")
#define OMP_FOR _Pragma("omp for")
#define OMP_SINGLE _Pragma("omp single")
#define OMP_SECTIONS _Pragma("omp sections")
#define OMP_SECTION _Pragma("omp section")
#endif
#else
#include <ctime>
#define OMP_PARALLEL
#define OMP_FOR
#define OMP_SINGLE
#define OMP_SECTIONS
#define OMP_SECTION
#endif
#include <cassert>
#include <vector>
#include "Types.h"

#include <string.h>
#include <memory>
#define hfid0 0
#define hfid1 1
namespace geodesic{

template<class T>			//quickly allocates multiple elements of a given type; no deallocation
class SimlpeMemoryAllocator
{
public:
	typedef T* pointer;

	SimlpeMemoryAllocator(unsigned block_size = 0, 
						  unsigned max_number_of_blocks = 0)
	{
		reset(block_size, 
			  max_number_of_blocks);
	};

	~SimlpeMemoryAllocator(){};

	void reset(unsigned block_size, 
			   unsigned max_number_of_blocks)
	{
		m_block_size = block_size;
		m_max_number_of_blocks = max_number_of_blocks;


		m_current_position = 0;

		m_storage.reserve(max_number_of_blocks);
		m_storage.resize(1);
		m_storage[0].resize(block_size);
	};

	pointer allocate(unsigned const n)		//allocate n units
	{
		assert(n < m_block_size);

		if(m_current_position + n >= m_block_size)
		{
			m_storage.push_back( std::vector<T>() );
			m_storage.back().resize(m_block_size);
			m_current_position = 0;
		}
		pointer result = & m_storage.back()[m_current_position];
		m_current_position += n;

		return result;
	};
private:
	std::vector<std::vector<T> > m_storage;	
	unsigned m_block_size;				//size of a single block
	unsigned m_max_number_of_blocks;		//maximum allowed number of blocks
	unsigned m_current_position;			//first unused element inside the current block
};


template<class T>		//quickly allocates and deallocates single elements of a given type
class MemoryAllocator
{
public:
	typedef T* pointer;

	MemoryAllocator(unsigned block_size = 1024, 
				    unsigned max_number_of_blocks = 1024)
	{
		reset(block_size, 
			  max_number_of_blocks);
	};

	~MemoryAllocator(){};

	void clear()
	{
		reset(m_block_size, 
			  m_max_number_of_blocks);
	}

	void reset(unsigned block_size, 
			   unsigned max_number_of_blocks)
	{
		m_block_size = block_size;
		m_max_number_of_blocks = max_number_of_blocks;

		assert(m_block_size > 0);
		assert(m_max_number_of_blocks > 0);

		m_current_position = 0;

		m_storage.reserve(max_number_of_blocks);
		m_storage.resize(1);
		m_storage[0].resize(block_size);

		m_deleted.clear();
		m_deleted.reserve(2*block_size);
	};

	pointer allocate()		//allocates single unit of memory
	{
		pointer result;
		if(m_deleted.empty())
		{
			if(m_current_position + 1 >= m_block_size)
			{
				m_storage.push_back( std::vector<T>() );
				m_storage.back().resize(m_block_size);
				m_current_position = 0;
			}
			result = & m_storage.back()[m_current_position];
			++m_current_position;
		}
		else
		{
			result = m_deleted.back();
			m_deleted.pop_back();
		}

		return result;
	};

	void deallocate(pointer p)		//allocate n units
	{
		if(m_deleted.size() < m_deleted.capacity())
		{
			m_deleted.push_back(p);
		}
	};

private:
	std::vector<std::vector<T> > m_storage;
	unsigned m_block_size;				//size of a single block
	unsigned m_max_number_of_blocks;		//maximum allowed number of blocks
	unsigned m_current_position;			//first unused element inside the current block

	std::vector<pointer> m_deleted;			//pointers to deleted elemets
};


class OutputBuffer
{
public:
	OutputBuffer():
		m_num_bytes(0)
	{}

	void clear()
	{
		m_num_bytes = 0;
        m_buffer = nullptr;
	}

	template<class T>
	T* allocate(unsigned n)
	{
        Scalar wanted = n*sizeof(T);
		if(wanted > m_num_bytes)
		{
            unsigned new_size = static_cast<unsigned>(ceil(wanted / static_cast<Scalar>(sizeof(Scalar))));
            m_buffer = std::make_unique<Scalar[]>(new_size);
            m_num_bytes = new_size * sizeof(Scalar);

		}

		return (T*)m_buffer.get();
	}

	template <class T>
	T* get()
	{
		return (T*)m_buffer.get();
	}

	template<class T>
	unsigned capacity()
	{
        return (unsigned)floor((Scalar)m_num_bytes/(Scalar)sizeof(T));
	};

private:

    std::unique_ptr<Scalar[]> m_buffer;
	unsigned m_num_bytes;
};
} 
namespace geodesic {

	class Interval;
	class IntervalList;
	typedef Interval* interval_pointer;
    typedef IntervalList* list_pointer;
    typedef OpenMesh::FaceHandle face_pointer;
    typedef OpenMesh::EdgeHandle edge_pointer;
    typedef OpenMesh::VertexHandle vertex_pointer;
    typedef OpenMesh::HalfedgeHandle halfedge_handle;

	struct Triangle // Components of a face to be propagated
	{
        face_pointer face; // Current Face

        edge_pointer bottom_edge, // Edges
			left_edge,
			right_edge;

        vertex_pointer top_vertex, // Vertices
			left_vertex,
			right_vertex;

        Scalar top_alpha,
			left_alpha,
			right_alpha; // Angles

		list_pointer left_list,
			right_list; // Lists
	};

	class Interval						//interval of the edge
	{
	public:

		Interval() {};
		~Interval() {};

        Scalar& start() { return m_start; };
        Scalar& stop() { return m_stop; };
        Scalar& d() { return m_d; };
        Scalar& pseudo_x() { return m_pseudo_x; };
        Scalar& pseudo_y() { return m_pseudo_y; };

        Scalar& sp() { return m_sp; };

        Scalar& shortest_distance() { return m_shortest_distance; }

		interval_pointer& next() { return m_next; };
		interval_pointer& previous() { return m_previous; };

	private:
        Scalar m_start;						//initial point of the interval on the edge
        Scalar m_stop;
        Scalar m_d;							//distance from the source to the pseudo-source
        Scalar m_pseudo_x;					//coordinates of the pseudo-source in the local coordinate system
        Scalar m_pseudo_y;					//y-coordinate should be always negative

        Scalar m_sp;                        //separating point

        Scalar m_shortest_distance;         //shortest distance from the interval to top_vertex, for numerical precision issue

		interval_pointer m_next;			//pointer to the next interval in the list	
		interval_pointer m_previous;        //pointer to the previous interval in the list
	};

	class IntervalList						//list of the of intervals of the given edge
	{
	public:
        IntervalList() {  /*m_start = NULL; m_edge = NULL;*/ m_sp = -1; m_begin = m_end = NULL; }
		~IntervalList() {};

		void clear() { m_begin = m_end = NULL; }
        void initialize(edge_pointer e) { m_edge = e; }

        vertex_pointer& start_vertex() { return m_start; }
        edge_pointer& edge() { return m_edge; }

        Scalar& sp() { return m_sp; };

        Scalar& pseudo_x() { return m_pseudo_x; };
        Scalar& pseudo_y() { return m_pseudo_y; };

		// List operation
		interval_pointer& begin() { return m_begin; }

		interval_pointer& end() { return m_end; }

		bool empty() { return m_begin == NULL; }

		void push_back(interval_pointer & w)
		{
			if (!m_end)
			{
				w->previous() = NULL;
				w->next() = NULL;
				m_begin = m_end = w;
			}
			else
			{
				w->next() = NULL;
				w->previous() = m_end;
				m_end->next() = w;
				m_end = w;
			}
		}

		void erase(interval_pointer & w)
		{
			if ((w == m_begin) && (w == m_end))
			{
				m_begin = m_end = NULL;
			}
			else if (w == m_begin)
			{
				m_begin = m_begin->next();
				m_begin->previous() = NULL;
			}
			else if (w == m_end)
			{
				m_end = m_end->previous();
				m_end->next() = NULL;
			}
			else
			{
				w->previous()->next() = w->next();
				w->next()->previous() = w->previous();
			}
		}

	private:

        edge_pointer     m_edge;		    //edge that owns this list
        vertex_pointer   m_start;           //vertex from which the interval list starts

		interval_pointer m_begin;
		interval_pointer m_end;

        Scalar m_pseudo_x;
        Scalar m_pseudo_y;

        Scalar m_sp;                        //separating point
	};

}		//geodesic
namespace geodesic{

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

Scalar const GEODESIC_INF = 1e100;

// statistics
inline Scalar m_time_consumed;		//how much time does the propagation step takes
inline unsigned m_queue_max_size;			//used for statistics
inline unsigned m_windows_propagation; // how many time a window is propagated
inline unsigned m_windows_wavefront; // the number of windows on the wavefront
inline unsigned m_windows_peak; // the maximum number of windows, used to calculate the memory

// two windows' states after checking
enum windows_state
{
	w1_invalid,
	w2_invalid,
	both_valid
};

inline Scalar cos_from_edges(Scalar const a,			//compute the cosine of the angle given the lengths of the edges
                             Scalar const b,
                             Scalar const c)
{
	assert(a > 1e-50);
	assert(b > 1e-50);
	assert(c > 1e-50);

    Scalar result = (b * b + c * c - a * a)/(2.0 * b * c);
    result = result>-1.0?result:-1.0;
    return result <1.0 ? result:1.0;
}

inline Scalar angle_from_edges(Scalar const a,			//compute the cosine of the angle given the lengths of the edges
                               Scalar const b,
                               Scalar const c)
{
	return acos(cos_from_edges(a, b, c));
}

template<class Points, class Faces>
inline bool read_mesh_from_file(char* filename,
								Points& points,
								Faces& faces, 
								std::vector<int> &realIndex, 
								int& originalVertNum)
{
	std::ifstream file(filename);
	assert(file.is_open());
	if(!file.is_open()) return false;

	char type;
	std::string curLine;
    Scalar coord[3];
	unsigned int vtxIdx[3];
	std::map<std::string, int> mapForDuplicate;
	originalVertNum = 0;

	while(getline(file, curLine))
	{
		if (curLine.size() < 2) continue;
		if (curLine[0] == 'v' && curLine[1] != 't')
		{
			std::map<std::string, int>::iterator pos = mapForDuplicate.find(curLine);
			if (pos == mapForDuplicate.end())
			{
				int oldSize = mapForDuplicate.size();
				realIndex.push_back(oldSize);
				mapForDuplicate[curLine] = oldSize;
                sscanf(curLine.c_str(), "v %lf %lf %lf", &coord[0], &coord[1], &coord[2]);
				for (int i = 0;i < 3;++i) points.push_back(coord[i]);
			}
			else
			{
				realIndex.push_back(pos->second);
			}
			++originalVertNum;
		}
		else if (curLine[0] == 'f')
		{
			unsigned tex;
			if (curLine.find('/') != std::string::npos)
                sscanf(curLine.c_str(), "f %d/%d %d/%d %d/%d", &vtxIdx[0], &tex, &vtxIdx[1], &tex, &vtxIdx[2], &tex);
			else
                sscanf(curLine.c_str(), "f %d %d %d", &vtxIdx[0], &vtxIdx[1], &vtxIdx[2]);
			
			vtxIdx[0] = realIndex[vtxIdx[0]-1];
			vtxIdx[1] = realIndex[vtxIdx[1]-1];
			vtxIdx[2] = realIndex[vtxIdx[2]-1];
			if (vtxIdx[0] == vtxIdx[1] || vtxIdx[0] == vtxIdx[2] || vtxIdx[1] == vtxIdx[2]) continue;

			for (int i = 0;i < 3;++i) faces.push_back(vtxIdx[i]);
		}
	}
	file.close();

	printf("There are %d non-coincidence vertices.\n", points.size() / 3);

	return true;
}



inline void CalculateIntersectionPoint(Scalar X1, Scalar Y1, // compute intersection point of two windows
    Scalar X2, Scalar Y2,
    Scalar X3, Scalar Y3,
    Scalar X4, Scalar Y4,
    Scalar &Intersect_X, Scalar &Intersect_Y)
{
    Scalar A1, B1, C1, A2, B2, C2;
	A1 = Y2 - Y1;
	B1 = X1 - X2;
	C1 = X2 * Y1 - X1 * Y2;
	A2 = Y4 - Y3;
	B2 = X3 - X4;
	C2 = X4 * Y3 - X3 * Y4;

	Intersect_X = (B2 * C1 - B1 * C2) / (A2 * B1 - A1 * B2);
	Intersect_Y = (A1 * C2 - A2 * C1) / (A2 * B1 - A1 * B2);
}


inline bool PointInTriangle(Scalar &X, Scalar &Y, // judge if a point is inside a triangle
    //Scalar Ax, Scalar Ay, // 0, 0
    Scalar &Bx, //Scalar By, // By = 0
    Scalar &Cx, Scalar &Cy)
{
    Scalar dot00 = Cx * Cx + Cy * Cy;// dot00 = dot(v0, v0)
    Scalar dot01 = Cx * Bx;// dot01 = dot(v0, v1)
    Scalar dot02 = Cx * X + Cy * Y;// dot02 = dot(v0, v2)
    Scalar dot11 = Bx * Bx; // dot11 = dot(v1, v1)
    Scalar dot12 = Bx * X;  // dot12 = dot(v1, v2)

    Scalar invDenom = 1 / (dot00 * dot11 - dot01 * dot01);
    Scalar u = (dot11 * dot02 - dot01 * dot12) * invDenom;
    Scalar v = (dot00 * dot12 - dot01 * dot02) * invDenom;

	//return (u >= 0) && (v >= 0) && (u + v < 1);
	return (u >= 1e-10) && (v >= 1e-10) && (u + v < 1 - 1e-10);
}


} //geodesic
namespace geodesic{
class GeodesicAlgorithmBase
{
public:
    vertex_pointer opposite_vertex(edge_pointer e, vertex_pointer v)
    {
        halfedge_handle hf0 = this->mesh()->halfedge_handle(e, hfid0);
        if(this->mesh()->from_vertex_handle(hf0).idx()==v.idx())
            return this->mesh()->to_vertex_handle(hf0);
        else
            return this->mesh()->from_vertex_handle(hf0);
    };

    vertex_pointer opposite_vertex(face_pointer f, edge_pointer e)
    {
        halfedge_handle hf = this->mesh()->halfedge_handle(e, hfid0);
        hf = this->mesh()->face_handle(hf)==f? hf : this->mesh()->opposite_halfedge_handle(hf);
        return this->mesh()->to_vertex_handle(this->mesh()->next_halfedge_handle(hf));
    };

    bool belongs_v(edge_pointer e, vertex_pointer v)
    {
        halfedge_handle hf = this->mesh()->halfedge_handle(e, hfid0);
        return this->mesh()->from_vertex_handle(hf) == v ||
                this->mesh()->to_vertex_handle(hf) == v;
    }

    edge_pointer next_edge(face_pointer f, edge_pointer e, vertex_pointer v)
    {

        halfedge_handle hf = this->mesh()->halfedge_handle(e, hfid0);
        hf = this->mesh()->face_handle(hf)==f? hf : this->mesh()->opposite_halfedge_handle(hf);
        halfedge_handle next_hf;
        if(this->mesh()->from_vertex_handle(hf)==v)
        {
            next_hf = this->mesh()->next_halfedge_handle(this->mesh()->next_halfedge_handle(hf));
        }
        else if(this->mesh()->to_vertex_handle(hf)==v)
            next_hf = this->mesh()->next_halfedge_handle(hf);
        else
        {
            std::cout << "next_edge e and v has no connection" << std::endl;
            exit(1);
        }
        return this->mesh()->edge_handle(next_hf);
    };

    Scalar vertex_angle(face_pointer f, vertex_pointer v)
    {
        halfedge_handle hf0 = this->mesh()->halfedge_handle(f);
        vertex_pointer v0 = this->mesh()->from_vertex_handle(hf0);
        vertex_pointer v1 = this->mesh()->to_vertex_handle(hf0);
        vertex_pointer v2 = this->mesh()->to_vertex_handle(this->mesh()->next_halfedge_handle(hf0));
        Scalar l1 = (this->mesh()->point(v0) - this->mesh()->point(v1)).norm();
        Scalar l2 = (this->mesh()->point(v1) - this->mesh()->point(v2)).norm();
        Scalar l3 = (this->mesh()->point(v2) - this->mesh()->point(v0)).norm();
        if(v0.idx()==v.idx())
        {
            return angle_from_edges(l2, l3, l1);
        }
        if(v1.idx()==v.idx())
        {
            return angle_from_edges(l3, l1, l2);
        }
        if(v2.idx()==v.idx())
        {
            return angle_from_edges(l1, l2, l3);
        }
    };

    void update_edgelen()
    {
        for(auto eit = this->mesh()->edges_begin(); eit != this->mesh()->edges_end(); eit++)
        {
            halfedge_handle hf = this->mesh()->halfedge_handle(*eit, hfid0);
            Vec3 v1 = this->mesh()->point(this->mesh()->from_vertex_handle(hf));
            Vec3 v2 = this->mesh()->point(this->mesh()->to_vertex_handle(hf));
            this->mesh()->data(*eit).length = (v1-v2).norm();
        }
    };

    void build_adjacencies();

    edge_pointer opposite_edge(face_pointer f, vertex_pointer v)
    {
        for(auto e_it = this->mesh()->fe_begin(f); e_it != this->mesh()->fe_end(f); ++e_it)
        {
            edge_pointer e = *e_it;
            if(!belongs_v(e, v))
            {
                return e;
            }
        }
    };

    face_pointer opposite_face(edge_pointer e, face_pointer f)
    {
        halfedge_handle hf = this->mesh()->halfedge_handle(e, hfid0);
        return this->mesh()->face_handle(hf) == f?
                    this->mesh()->face_handle(this->mesh()->opposite_halfedge_handle(hf))
                  :this->mesh()->face_handle(hf);
    };

    void initialize(Mesh* mesh);

    GeodesicAlgorithmBase(Mesh* mesh)
    {
       initialize(mesh);
    };

    GeodesicAlgorithmBase(){m_mesh = NULL;};

    virtual ~GeodesicAlgorithmBase(){};

    virtual void print_statistics()		//print info about timing and memory usage in the propagation step of the algorithm
    {
        std::cout << "propagation step took " << m_time_consumed << " seconds " << std::endl;
    };

    Mesh* mesh(){return m_mesh;};

    // propagate a window
    bool compute_propagated_parameters(Scalar pseudo_x,
                                       Scalar pseudo_y,
                                       Scalar start,
                                       Scalar end,		//start/end of the interval
                                       Scalar alpha,	//corner angle
                                       Scalar L,		//length of the new edge
                                       interval_pointer candidates,
                                       Scalar d);		//if it is the last interval on the edge

    // intersection point on an edge
    Scalar compute_positive_intersection(Scalar start,
                                         Scalar pseudo_x,
                                         Scalar pseudo_y,
                                         Scalar sin_alpha,
                                         Scalar cos_alpha);

    inline bool calculate_triangle_parameters(list_pointer &list, Triangle &Tri); // calculate the parameters of the triangle to be propagated

    list_pointer interval_list_0(edge_pointer e)
    {
        return &m_edge_interval_lists_0[e.idx()];
    };

    list_pointer interval_list_1(edge_pointer e)
    {
        return &m_edge_interval_lists_1[e.idx()];
    };


protected:

    Mesh* m_mesh;

    Triangle Tri; // the triangle to be propagated

    IntervalList wl_left, wl_right;

    std::vector<IntervalList> m_edge_interval_lists_0;		// windows propagated from adjacent_face[0] of the edge
    std::vector<IntervalList> m_edge_interval_lists_1;		// windows propagated from adjacent_face[1] of the edge

};





















inline void GeodesicAlgorithmBase::initialize(Mesh* mesh)
{
    m_mesh = mesh;
    m_edge_interval_lists_0.resize(mesh->n_edges());
    m_edge_interval_lists_1.resize(mesh->n_edges());

    // initialize statistics
    m_queue_max_size      = 0;
    m_windows_propagation = 0;
    m_windows_wavefront   = 0;
    m_windows_peak        = 0;

    // initialize window lists, similar to half-edge structure
    for (unsigned i = 0; i < m_edge_interval_lists_0.size(); ++i)
    {
        edge_pointer edge = mesh->edge_handle(i);
        m_edge_interval_lists_0[i].initialize(edge);
        m_edge_interval_lists_1[i].initialize(edge);
        interval_list_0(edge)->start_vertex() = mesh->from_vertex_handle(mesh->halfedge_handle(edge, hfid0));
        interval_list_1(edge)->start_vertex() = mesh->from_vertex_handle(mesh->halfedge_handle(edge, hfid1));
    }
    // verify list links
    for (unsigned i = 0; i < mesh->n_faces(); ++i)
    {
        face_pointer f = (mesh->face_handle(i));
        vertex_pointer v[3];
        size_t j = 0;
        for (auto e_it = mesh->fe_begin(f); e_it != mesh->fe_end(f); ++e_it)
        {
            edge_pointer e = *e_it;
            if (mesh->face_handle(mesh->halfedge_handle(e, hfid0)) == f)
                v[j] = interval_list_0(e)->start_vertex();
            else
                v[j] = interval_list_1(e)->start_vertex();

            if ((interval_list_0(e)->start_vertex().idx() < 0) || (interval_list_1(e)->start_vertex().idx() < 0))
            {
                std::cout << "list link error" << std::endl;
                exit(1);
            }

            if (interval_list_0(e)->start_vertex() == interval_list_1(e)->start_vertex())
            {
                std::cout << "list link error" << std::endl;
                exit(1);
            }

            if (!((belongs_v(e, interval_list_0(e)->start_vertex())) &&
                  (belongs_v(e, interval_list_1(e)->start_vertex()))))
            {
                std::cout << "list link error" << std::endl;
                exit(1);
            }
            j++;
        }
        if ((v[0].idx() >-1 && v[0] == v[1]) || (v[0].idx() >-1 && v[0] == v[2]) || (v[1].idx() >-1 && v[1] == v[2]))
        {
            std::cout << "list link error" << std::endl;
            exit(1);
        }
    }
    update_edgelen();
    build_adjacencies();
};

inline Scalar GeodesicAlgorithmBase::compute_positive_intersection(Scalar start,
                                                                   Scalar pseudo_x,
                                                                   Scalar pseudo_y,
                                                                   Scalar sin_alpha,
                                                                   Scalar cos_alpha)
{
    //assert(pseudo_y < 0);
    assert(pseudo_y <= 0);

    Scalar denominator = sin_alpha*(pseudo_x - start) - cos_alpha*pseudo_y;
    if (denominator < 0.0)
    {
        return -1.0;
    }

    Scalar numerator = -pseudo_y*start;

    if (numerator < 1e-30)
    {
        return 0.0;
    }

    if (denominator < 1e-30)
    {
        return -1.0;
    }

    return numerator / denominator;
}

inline bool GeodesicAlgorithmBase::compute_propagated_parameters(Scalar pseudo_x,
                                                                 Scalar pseudo_y,
                                                                 Scalar begin,
                                                                 Scalar end,		//start/end of the interval
                                                                 Scalar alpha,	//corner angle
                                                                 Scalar L,		//length of the new edge
                                                                 interval_pointer candidates,
                                                                 Scalar d)
{
    assert(pseudo_y <= 0.0);
    assert(begin <= end);
    assert(begin >= 0);

    ++m_windows_propagation; // Statistics

    interval_pointer p = candidates;

    Scalar sin_alpha = sin(alpha);
    Scalar cos_alpha = cos(alpha);

    //important: for the first_interval, this function returns zero only if the new edge is "visible" from the source
    //if the new edge can be covered only after turn_over, the value is negative (-1.0)
    Scalar L1 = compute_positive_intersection(begin,
                                              pseudo_x,
                                              pseudo_y,
                                              sin_alpha,
                                              cos_alpha);

    if (L1 < 0 || L1 >= L) // Does not produce a window on the edge
        return false;

    Scalar L2 = compute_positive_intersection(end,
                                              pseudo_x,
                                              pseudo_y,
                                              sin_alpha,
                                              cos_alpha);

    if (L2 < 0 || L2 >= L) // Covers vertex
    {
        p->start() = L1;
        p->stop() = L;
        p->pseudo_x() = cos_alpha*pseudo_x + sin_alpha*pseudo_y;
        p->pseudo_y() = -sin_alpha*pseudo_x + cos_alpha*pseudo_y;
        assert(p->pseudo_y() <= 0.0);

        return true;
    }
    else
    {
        // Does not cover vertex
        p->start() = L1;
        p->stop() = L2;
        p->pseudo_x() = cos_alpha*pseudo_x + sin_alpha*pseudo_y;
        p->pseudo_y() = -sin_alpha*pseudo_x + cos_alpha*pseudo_y;
        assert(p->pseudo_y() <= 0.0);

        return true;
    }
}

inline bool GeodesicAlgorithmBase::calculate_triangle_parameters(list_pointer &list, Triangle &Tri) // Calculate the parameters of the triangle to be propagated
{
    OpenMesh::HalfedgeHandle hf0 = this->mesh()->halfedge_handle(list->edge(), hfid0);
    OpenMesh::HalfedgeHandle hf1 = this->mesh()->halfedge_handle(list->edge(), hfid1);
    size_t adjface_size=0;
    if(this->mesh()->face_handle(hf0).idx()>-1)
        adjface_size++;
    if(this->mesh()->face_handle(hf1).idx()>-1)
        adjface_size++;

    if (adjface_size > 1)
    {
        Tri.bottom_edge = list->edge();

        if (list == interval_list_0(Tri.bottom_edge))
            Tri.face = this->mesh()->face_handle(hf1);
        else
            Tri.face = this->mesh()->face_handle(hf0);

        Tri.top_vertex = opposite_vertex(Tri.face, Tri.bottom_edge);
        Tri.left_vertex = list->start_vertex();
        Tri.right_vertex = opposite_vertex(Tri.bottom_edge, Tri.left_vertex);

        Tri.left_edge = next_edge(Tri.face, Tri.bottom_edge, Tri.left_vertex);
        Tri.right_edge = next_edge(Tri.face, Tri.bottom_edge, Tri.right_vertex);

        Tri.top_alpha = vertex_angle(Tri.face, Tri.top_vertex);
        Tri.left_alpha = vertex_angle(Tri.face, Tri.left_vertex);
        Tri.right_alpha = vertex_angle(Tri.face, Tri.right_vertex);

        if (this->mesh()->face_handle(this->mesh()->halfedge_handle(Tri.left_edge, hfid0)) == Tri.face)
            Tri.left_list = interval_list_0(Tri.left_edge);
        else
            Tri.left_list = interval_list_1(Tri.left_edge);

        if (this->mesh()->face_handle(this->mesh()->halfedge_handle(Tri.right_edge, hfid0)) == Tri.face)
            Tri.right_list = interval_list_0(Tri.right_edge);
        else
            Tri.right_list = interval_list_1(Tri.right_edge);

        return false;
    }
    else
    {
        return true;
    }
}


inline void GeodesicAlgorithmBase::build_adjacencies()
{
    // define m_turn_around_flag for vertices
    std::vector<Scalar> total_vertex_angle(this->mesh()->n_vertices(), 0);
    for(auto f_it = this->mesh()->faces_begin(); f_it != this->mesh()->faces_end(); ++f_it)
    {
        halfedge_handle hf0 = this->mesh()->halfedge_handle(*f_it);
        vertex_pointer v0 = this->mesh()->from_vertex_handle(hf0);
        vertex_pointer v1 = this->mesh()->to_vertex_handle(hf0);
        vertex_pointer v2 = this->mesh()->to_vertex_handle(this->mesh()->next_halfedge_handle(hf0));
        Scalar l1 = (this->mesh()->point(v0) - this->mesh()->point(v1)).norm();
        Scalar l2 = (this->mesh()->point(v1) - this->mesh()->point(v2)).norm();
        Scalar l3 = (this->mesh()->point(v2) - this->mesh()->point(v0)).norm();

        total_vertex_angle[v0.idx()] += angle_from_edges(l2, l3, l1);
        total_vertex_angle[v1.idx()] += angle_from_edges(l3, l1, l2);
        total_vertex_angle[v2.idx()] += angle_from_edges(l1, l2, l3);
    }

    for(auto v_it = this->mesh()->vertices_begin(); v_it != this->mesh()->vertices_end(); ++v_it)
    {
        vertex_pointer v = *v_it;
        this->mesh()->data(v).saddle_or_boundary = (total_vertex_angle[v.idx()] > 2.0*M_PI - 1e-5);
    }

    for(auto e_it = this->mesh()->edges_begin(); e_it != this->mesh()->edges_end(); ++e_it)
    {
        edge_pointer e = *e_it;
        if(this->mesh()->is_boundary(e))
        {
            halfedge_handle hf = this->mesh()->halfedge_handle(e, hfid0);
            this->mesh()->data(this->mesh()->from_vertex_handle(hf)).saddle_or_boundary = true;
            this->mesh()->data(this->mesh()->to_vertex_handle(hf)).saddle_or_boundary = true;
        }
    }
}


}
namespace geodesic {

	class GeodesicAlgorithmExact : public GeodesicAlgorithmBase
	{
	public:

		// basic functions related to class
        GeodesicAlgorithmExact(Mesh* mesh) :
			GeodesicAlgorithmBase(mesh),
            m_memory_allocator(mesh->n_edges(), mesh->n_edges())
        {ori_mesh = mesh;};

        // construct neighboring mesh around src within m_radius, if no mesh is constructed,
        // increase m_radius with increase_radio
        GeodesicAlgorithmExact(Mesh* mesh, size_t src, Scalar m_radius)
        {
            ori_mesh = mesh;
            Mesh* sub_mesh = new Mesh;
            bool ok = construct_submesh(sub_mesh, src, m_radius);
            if(ok)
            {
                GeodesicAlgorithmBase::initialize(sub_mesh);
                m_memory_allocator.reset(sub_mesh->n_edges(), sub_mesh->n_edges());
            }
            else
            {
                std::cerr << "Error:Some points cannot be covered under the specified radius, please increase the radius" << std::endl;
                exit(1);
            }
        };
		~GeodesicAlgorithmExact() {};
        void clear() {
            m_memory_allocator.clear();
            for(auto v_it = this->mesh()->vertices_begin();v_it != this->mesh()->vertices_end(); ++v_it)
            {
                this->mesh()->data(*v_it).geodesic_distance = GEODESIC_INF;
            }
       };

        // main entry
        void propagate(unsigned source, std::vector<size_t>& idxs);

        // print the resulting statistics
        void print_statistics();
	private:

        // simple functions
        void initialize_propagation_data();
        void create_pseudo_source_windows(vertex_pointer &v, bool UpdateFIFOQueue);
        void erase_from_queue(vertex_pointer& v);

        // propagate a windows list (Rule 1)
        void find_separating_point(list_pointer &list); // find the separating point of the windows and the list
        void propagate_windows_to_two_edges(list_pointer &list); // propagates windows to two edges accross a triangle face

        // pairwise windows checking (Rule 2)
        void check_with_vertices(list_pointer &list);
        windows_state check_between_two_windows(interval_pointer &w1, interval_pointer &w2); // Check two neighbouring crossing windows on same edge
        void pairwise_windows_checking(list_pointer &list); // Check crossing windows on same edge

        // main operation
        void propagate_one_windows_list(list_pointer &list);

        // construct neighboring mesh
        bool construct_submesh(Mesh* sub_mesh, size_t source_idx, Scalar radius);

		// member variables
        std::set<vertex_pointer> m_vertex_queue;
		std::queue<list_pointer> m_list_queue;                // FIFO queue for lists
		MemoryAllocator<Interval> m_memory_allocator;		  // quickly allocate and deallocate intervals 
        Scalar neighbor_radius;

        Eigen::VectorXi SubVidxfromMesh;
        std::vector<int> MeshVidxfromSub;

        Mesh* ori_mesh;
		unsigned m_source;
	};


	//----------------- simple functions ---------------------
	inline void GeodesicAlgorithmExact::initialize_propagation_data()
	{
		clear();

		// initialize source's parameters
        vertex_pointer source = (this->mesh()->vertex_handle(m_source));
        this->mesh()->data(source).geodesic_distance = 0;
        this->mesh()->data(source).state = VertexState::INSIDE;

		// initialize windows around source
		create_pseudo_source_windows(source, false);
	}

    inline void GeodesicAlgorithmExact::erase_from_queue(vertex_pointer& v)
	{
        assert(m_vertex_queue.count(v) <= 1);

        std::multiset<vertex_pointer>::iterator it = m_vertex_queue.find(v);
        if (it != m_vertex_queue.end())
            m_vertex_queue.erase(it);
	}

	inline void GeodesicAlgorithmExact::create_pseudo_source_windows(vertex_pointer &pseudo_source, bool inside_traversed_area)
	{
		// update vertices around pseudo_source
        for (auto e_it = this->mesh()->ve_begin(pseudo_source); e_it != this->mesh()->ve_end(pseudo_source); ++e_it)
		{
            edge_pointer   edge_it = *e_it;
            vertex_pointer vert_it = opposite_vertex(edge_it, pseudo_source);

            Scalar distance = this->mesh()->data(pseudo_source).geodesic_distance
                    + this->mesh()->data(edge_it).length;

            if (distance < this->mesh()->data(vert_it).geodesic_distance)
			{
				m_vertex_queue.erase(vert_it);

                this->mesh()->data(vert_it).geodesic_distance = distance;
                if (this->mesh()->data(vert_it).state == VertexState::OUTSIDE)
                    this->mesh()->data(vert_it).state = VertexState::FRONT;

                this->mesh()->data(vert_it).incident_face = this->mesh()->face_handle(this->mesh()->halfedge_handle(edge_it, hfid0));
                edge_pointer next_edge = geodesic::GeodesicAlgorithmBase::next_edge(
                            this->mesh()->data(vert_it).incident_face,edge_it, pseudo_source);
                this->mesh()->data(vert_it).incident_point =
                        (this->mesh()->from_vertex_handle(this->mesh()->halfedge_handle(next_edge, hfid0)) == pseudo_source) ?
                            0 : this->mesh()->data(next_edge).length;

                m_vertex_queue.insert(vert_it);
			}
		}

        // update pseudo_source windows around pseudo_source
        for(auto f_it = this->mesh()->vf_begin(pseudo_source); f_it != this->mesh()->vf_end(pseudo_source); ++f_it)
		{
            face_pointer face_it = *f_it;
            edge_pointer edge_it = geodesic::GeodesicAlgorithmBase::opposite_edge(face_it, pseudo_source);
            list_pointer list = (this->mesh()->face_handle(this->mesh()->halfedge_handle(edge_it, hfid0))==face_it)?
                    interval_list_0(edge_it) : interval_list_1(edge_it);

			// create a window
			interval_pointer candidate = new Interval;

			candidate->start() = 0;
            candidate->stop() = this->mesh()->data(edge_it).length;
            candidate->d() = this->mesh()->data(pseudo_source).geodesic_distance;
            Scalar angle = geodesic::GeodesicAlgorithmBase::vertex_angle(face_it, list->start_vertex());
            Scalar length = this->mesh()->data(geodesic::GeodesicAlgorithmBase::next_edge
                                               (face_it, edge_it,list->start_vertex())).length;
			candidate->pseudo_x() = cos(angle) * length;
			candidate->pseudo_y() = -sin(angle) * length;

			// insert into list
			list->push_back(candidate);

			// push into M_LIST_QUEUE if inside traversed area
            vertex_pointer v0 = this->mesh()->from_vertex_handle(this->mesh()->halfedge_handle(edge_it, hfid0));
            vertex_pointer v1 = this->mesh()->from_vertex_handle(this->mesh()->halfedge_handle(edge_it, hfid1));
            if ((inside_traversed_area) &&
                    ((this->mesh()->data(v0).state != VertexState::FRONT)
                     || (this->mesh()->data(v1).state != VertexState::FRONT)))
				m_list_queue.push(list);

			// Statistics
			++m_windows_wavefront;
			if (m_windows_peak < m_windows_wavefront)
				m_windows_peak = m_windows_wavefront;

		}
    }

	//----------------- propagate a windows list (Rule 1) ---------------------
	inline void GeodesicAlgorithmExact::find_separating_point(list_pointer &list)
    {
        const Scalar LOCAL_EPSILON = 1e-20 * this->mesh()->data(list->edge()).length; // numerical issue

        Scalar L = this->mesh()->data(Tri.left_edge).length;
        Scalar top_x = L * cos(Tri.left_alpha);
        Scalar top_y = L * sin(Tri.left_alpha);

        Scalar temp_geodesic = GEODESIC_INF;
        face_pointer temp_face_handle = this->mesh()->data(Tri.top_vertex).incident_face;
        Scalar temp_incident_point = this->mesh()->data(Tri.top_vertex).incident_point;

		interval_pointer iter = list->begin();

        Scalar wlist_sp = 0;
        Scalar wlist_pseudo_x = 0;
        Scalar wlist_pseudo_y = 0;

		while (iter != NULL)
		{
            interval_pointer &w = iter;

            Scalar w_sp = w->pseudo_x() - w->pseudo_y() * ((top_x - w->pseudo_x()) / (top_y - w->pseudo_y()));
            Scalar distance = GEODESIC_INF;

			// shortest path from the window
			if ((w_sp - w->start() > LOCAL_EPSILON) && (w_sp - w->stop() < -LOCAL_EPSILON))
			{
				distance = w->d() + sqrt((top_x - w->pseudo_x()) * (top_x - w->pseudo_x()) + (top_y - w->pseudo_y()) * (top_y - w->pseudo_y()));
				w->shortest_distance() = distance;
			}
			else if (w_sp - w->start() <= LOCAL_EPSILON)
			{
				distance = w->d() + sqrt((top_x - w->start()) * (top_x - w->start()) + top_y * top_y) + sqrt((w->start() - w->pseudo_x()) * (w->start() - w->pseudo_x()) + w->pseudo_y() * w->pseudo_y());
				w->shortest_distance() = distance;
				w_sp = w->start();
			}
			else if (w_sp - w->stop() >= -LOCAL_EPSILON)
			{
				distance = w->d() + sqrt((top_x - w->stop()) * (top_x - w->stop()) + top_y * top_y) + sqrt((w->stop() - w->pseudo_x()) * (w->stop() - w->pseudo_x()) + w->pseudo_y() * w->pseudo_y());
				w->shortest_distance() = distance;
				w_sp = w->stop();
			}

			// update information at top_t
            if (distance < temp_geodesic)
			{
                temp_geodesic = distance;
                temp_face_handle = Tri.face;
                vertex_pointer v0 = this->mesh()->from_vertex_handle(this->mesh()->halfedge_handle(list->edge(), hfid0));
                temp_incident_point = (list->start_vertex() == v0) ?
                            w_sp : this->mesh()->data(list->edge()).length - w_sp;
				wlist_sp = w_sp;
				wlist_pseudo_x = w->pseudo_x();
				wlist_pseudo_y = w->pseudo_y();
			}
			w->sp() = w_sp;

			iter = iter->next();
		}

        // update top_vertex and M_VERTEX_QUEUE
        if (temp_geodesic < this->mesh()->data(Tri.top_vertex).geodesic_distance)
		{
            if (this->mesh()->data(Tri.top_vertex).state == VertexState::FRONT) erase_from_queue(Tri.top_vertex);
            this->mesh()->data(Tri.top_vertex).geodesic_distance = temp_geodesic;
            this->mesh()->data(Tri.top_vertex).incident_face = temp_face_handle;
            this->mesh()->data(Tri.top_vertex).incident_point = temp_incident_point;
            if (this->mesh()->data(Tri.top_vertex).state == VertexState::FRONT)
            {
                m_vertex_queue.insert(Tri.top_vertex);
            }

            if ((this->mesh()->data(Tri.top_vertex).state == VertexState::INSIDE)
                    && (this->mesh()->data(Tri.top_vertex).saddle_or_boundary))
                create_pseudo_source_windows(Tri.top_vertex, true); // handle saddle vertex
		}

		list->sp() = wlist_sp;
		list->pseudo_x() = wlist_pseudo_x;
		list->pseudo_y() = wlist_pseudo_y;
	}

	inline void GeodesicAlgorithmExact::propagate_windows_to_two_edges(list_pointer &list)
	{
        const Scalar LOCAL_EPSILON = 1e-8 * this->mesh()->data(list->edge()).length; // numerical issue

		interval_pointer iter = list->begin();
		interval_pointer iter_t;

		enum PropagationDirection
		{
			LEFT,
			RIGHT,
			BOTH
		};

		PropagationDirection direction;

		while (!list->empty() && (iter != NULL))
		{
			interval_pointer &w = iter;
			assert(w->start() <= w->stop());

			if (w->sp() < list->sp() - LOCAL_EPSILON)
			{
				// only propagate to left edge
                Scalar Intersect_X, Intersect_Y;

				// judge the positions of the two windows
				CalculateIntersectionPoint(list->pseudo_x(), list->pseudo_y(), list->sp(), 0, w->pseudo_x(), w->pseudo_y(), w->stop(), 0, Intersect_X, Intersect_Y);
				if ((w->stop() < list->sp()) || ((Intersect_Y <= 0) && (Intersect_Y >= list->pseudo_y()) && (Intersect_Y >= w->pseudo_y())))
				{
					direction = PropagationDirection::LEFT;
				}
				else
				{
					direction = PropagationDirection::BOTH;
				}				
			}
			else if (w->sp() > list->sp() + LOCAL_EPSILON)
			{
				// only propagate to right edge
                Scalar Intersect_X, Intersect_Y;

				// judge the positions of the two windows
				CalculateIntersectionPoint(list->pseudo_x(), list->pseudo_y(), list->sp(), 0, w->pseudo_x(), w->pseudo_y(), w->start(), 0, Intersect_X, Intersect_Y);
				if ((w->start() > list->sp())||((Intersect_Y <= 0) && (Intersect_Y >= list->pseudo_y()) && (Intersect_Y >= w->pseudo_y())))
				{
					direction = PropagationDirection::RIGHT;
				}
				else
				{
					direction = PropagationDirection::BOTH;
				}	
			}
			else
			{
				// propagate to both edges
				direction = PropagationDirection::BOTH;
			}

			bool ValidPropagation;
			interval_pointer right_w;

			switch (direction) {
			case PropagationDirection::LEFT:
				ValidPropagation = compute_propagated_parameters(w->pseudo_x(),
					w->pseudo_y(),
					w->start(),
					w->stop(),
					Tri.left_alpha,
                    this->mesh()->data(Tri.left_edge).length,
					w,
					w->d());

				iter_t = iter->next();
				if (ValidPropagation)
				{
					list->erase(w);
					wl_left.push_back(w);
				}
				else
				{
					list->erase(w);
					delete w;
					--m_windows_wavefront;
				}
				iter = iter_t;
				break;

			case PropagationDirection::RIGHT:
                ValidPropagation = compute_propagated_parameters(this->mesh()->data(Tri.bottom_edge).length - w->pseudo_x(),
					w->pseudo_y(),
                    this->mesh()->data(Tri.bottom_edge).length - w->stop(),
                    this->mesh()->data(Tri.bottom_edge).length - w->start(),
					Tri.right_alpha,
                    this->mesh()->data(Tri.right_edge).length,
					w,
					w->d());

				iter_t = iter->next();
				if (ValidPropagation)
				{
                    Scalar length = this->mesh()->data(Tri.right_edge).length; // invert window
                    Scalar start = length - w->stop();
					w->stop() = length - w->start();
					w->start() = start;
					w->pseudo_x() = length - w->pseudo_x();

					list->erase(w);
					wl_right.push_back(w);
				}
				else
				{
					list->erase(w);
					delete w;
					--m_windows_wavefront;
				}
				iter = iter_t;
				break;

			case PropagationDirection:: BOTH:
				right_w = new Interval;
				memcpy(right_w, w, sizeof(Interval));

				ValidPropagation = compute_propagated_parameters(w->pseudo_x(),
					w->pseudo_y(),
					w->start(),
					w->stop(),
                    geodesic::GeodesicAlgorithmBase::vertex_angle(Tri.face, Tri.left_vertex),
                    this->mesh()->data(Tri.left_edge).length,
					w,
					w->d());

				iter_t = iter->next();
				if (ValidPropagation)
				{
					list->erase(w);
					wl_left.push_back(w);
				}
				else
				{
					list->erase(w);
					delete w;
					--m_windows_wavefront;
				}
				iter = iter_t;

                ValidPropagation = compute_propagated_parameters(this->mesh()->data(Tri.bottom_edge).length - right_w->pseudo_x(),
					right_w->pseudo_y(),
                    this->mesh()->data(Tri.bottom_edge).length - right_w->stop(),
                    this->mesh()->data(Tri.bottom_edge).length - right_w->start(),
                    geodesic::GeodesicAlgorithmBase::vertex_angle(Tri.face, Tri.right_vertex),
                    this->mesh()->data(Tri.right_edge).length,
					right_w,
					right_w->d());

				if (ValidPropagation)
				{
					// invert window
                    Scalar length = this->mesh()->data(Tri.right_edge).length;
                    Scalar start = length - right_w->stop();
					right_w->stop() = length - right_w->start();
					right_w->start() = start;
					right_w->pseudo_x() = length - right_w->pseudo_x();

					wl_right.push_back(right_w);

					++m_windows_wavefront;
					if (m_windows_peak < m_windows_wavefront)
						m_windows_peak = m_windows_wavefront;
				}
				else
				{
					delete right_w;
				}
				break;

			default:
				break;
			}
		}
	}

	//----------------- pairwise windows checking (Rule 2) ----------------------
	inline void GeodesicAlgorithmExact::check_with_vertices(list_pointer &list)
    {
		if (list->empty()) return;

		interval_pointer iter = list->begin();
		interval_pointer iter_t;

		while ((!list->empty()) && (iter != NULL))
		{
			interval_pointer &w = iter;
			bool w_survive = true;

            edge_pointer   e = list->edge();
			vertex_pointer v1 = list->start_vertex();
            vertex_pointer v2 = opposite_vertex(e, v1);
            Scalar d1 = GEODESIC_INF;

			d1 = w->d() + sqrt((w->stop() - w->pseudo_x()) * (w->stop() - w->pseudo_x()) + w->pseudo_y() * w->pseudo_y());
            if (this->mesh()->data(v1).geodesic_distance + w->stop() < d1)
				w_survive = false;

			d1 = w->d() + sqrt((w->start() - w->pseudo_x()) * (w->start() - w->pseudo_x()) + w->pseudo_y() * w->pseudo_y());
            if (this->mesh()->data(v2).geodesic_distance + this->mesh()->data(e).length - w->start() < d1)
				w_survive = false;


			iter_t = iter;
			iter = iter->next();

			if (!w_survive)
			{
				list->erase(iter_t);
				delete iter_t;
				--m_windows_wavefront;
			}
		}
	}

	inline windows_state GeodesicAlgorithmExact::check_between_two_windows(interval_pointer &w1, interval_pointer &w2)
	{
        Scalar NUMERCIAL_EPSILON = 1 - 1e-12;
		// we implement the discussed 6 cases as follows for simplicity

		if ((w1->start() >= w2->start()) && (w1->start() <= w2->stop())) // w1->start
		{
            Scalar Intersect_X, Intersect_Y;

			// judge the order of the two windows
			CalculateIntersectionPoint(w2->pseudo_x(), w2->pseudo_y(), w1->start(), 0, w1->pseudo_x(), w1->pseudo_y(), w1->stop(), 0, Intersect_X, Intersect_Y);

			if ((Intersect_Y <= 0) && (Intersect_Y >= w1->pseudo_y()) && (Intersect_Y >= w2->pseudo_y()))
			{
                Scalar d1, d2;
				d1 = w1->d() + sqrt((w1->start() - w1->pseudo_x()) * (w1->start() - w1->pseudo_x()) + (w1->pseudo_y()) * (w1->pseudo_y()));
				d2 = w2->d() + sqrt((w1->start() - w2->pseudo_x()) * (w1->start() - w2->pseudo_x()) + (w2->pseudo_y()) * (w2->pseudo_y()));

				if (d2 < d1 * NUMERCIAL_EPSILON)
					return w1_invalid;
				if (d1 < d2 * NUMERCIAL_EPSILON)
					w2->start() = w1->start();
			}
		}

		if ((w1->stop() >= w2->start()) && (w1->stop() <= w2->stop())) // w1->stop
		{
            Scalar Intersect_X, Intersect_Y;

			// judge the order of the two windows
			CalculateIntersectionPoint(w2->pseudo_x(), w2->pseudo_y(), w1->stop(), 0, w1->pseudo_x(), w1->pseudo_y(), w1->start(), 0, Intersect_X, Intersect_Y);

			if ((Intersect_Y <= 0) && (Intersect_Y >= w1->pseudo_y()) && (Intersect_Y >= w2->pseudo_y()))
			{
                Scalar d1, d2;
				d1 = w1->d() + sqrt((w1->stop() - w1->pseudo_x()) * (w1->stop() - w1->pseudo_x()) + (w1->pseudo_y()) * (w1->pseudo_y()));
				d2 = w2->d() + sqrt((w1->stop() - w2->pseudo_x()) * (w1->stop() - w2->pseudo_x()) + (w2->pseudo_y()) * (w2->pseudo_y()));

				if (d2 < d1 * NUMERCIAL_EPSILON)
					return w1_invalid;
				if (d1 < d2 * NUMERCIAL_EPSILON)
					w2->stop() = w1->stop();
			}
		}

		if ((w2->start() >= w1->start()) && (w2->start() <= w1->stop())) // w2->start
		{
            Scalar Intersect_X, Intersect_Y;

			// judge the previous order of the two windows
			CalculateIntersectionPoint(w1->pseudo_x(), w1->pseudo_y(), w2->start(), 0, w2->pseudo_x(), w2->pseudo_y(), w2->stop(), 0, Intersect_X, Intersect_Y);

			if ((Intersect_Y <= 0) && (Intersect_Y >= w1->pseudo_y()) && (Intersect_Y >= w2->pseudo_y()))
			{
                Scalar d1, d2;
				d1 = w1->d() + sqrt((w2->start() - w1->pseudo_x()) * (w2->start() - w1->pseudo_x()) + (w1->pseudo_y()) * (w1->pseudo_y()));
				d2 = w2->d() + sqrt((w2->start() - w2->pseudo_x()) * (w2->start() - w2->pseudo_x()) + (w2->pseudo_y()) * (w2->pseudo_y()));

				if (d1 < d2 * NUMERCIAL_EPSILON)
					return w2_invalid;
				if (d2 < d1 * NUMERCIAL_EPSILON)
					w1->start() = w2->start();
			}
		}

		if ((w2->stop() >= w1->start()) && (w2->stop() <= w1->stop())) // w2->stop
		{
            Scalar Intersect_X, Intersect_Y;

			// judge the previous order of the two windows
			CalculateIntersectionPoint(w1->pseudo_x(), w1->pseudo_y(), w2->stop(), 0, w2->pseudo_x(), w2->pseudo_y(), w2->start(), 0, Intersect_X, Intersect_Y);

			if ((Intersect_Y <= 0) && (Intersect_Y >= w1->pseudo_y()) && (Intersect_Y >= w2->pseudo_y()))
			{
                Scalar d1, d2;
				d1 = w1->d() + sqrt((w2->stop() - w1->pseudo_x()) * (w2->stop() - w1->pseudo_x()) + (w1->pseudo_y()) * (w1->pseudo_y()));
				d2 = w2->d() + sqrt((w2->stop() - w2->pseudo_x()) * (w2->stop() - w2->pseudo_x()) + (w2->pseudo_y()) * (w2->pseudo_y()));

				if (d1 < d2 * NUMERCIAL_EPSILON)
					return w2_invalid;
				if (d2 < d1 * NUMERCIAL_EPSILON)
					w1->stop() = w2->stop();
			}
		}

		if (w1->start() >= w2->stop())
		{
            Scalar Intersect_X, Intersect_Y;

			// judge the previous order of the two windows
			CalculateIntersectionPoint(w1->pseudo_x(), w1->pseudo_y(), w1->start(), 0, w2->pseudo_x(), w2->pseudo_y(), w2->stop(), 0, Intersect_X, Intersect_Y);

            face_pointer f = opposite_face(Tri.bottom_edge, Tri.face);
            edge_pointer e = next_edge(f, Tri.bottom_edge, Tri.left_vertex);
            Scalar angle = vertex_angle(f, Tri.left_vertex);
            Scalar Cx = this->mesh()->data(e).length * cos(angle);
            Scalar Cy = this->mesh()->data(e).length * -sin(angle);

            if ((PointInTriangle(Intersect_X, Intersect_Y, this->mesh()->data(Tri.bottom_edge).length, Cx, Cy))
				&& (Intersect_Y <= 0) && (Intersect_Y >= w1->pseudo_y()) && (Intersect_Y >= w2->pseudo_y()))
			{
                Scalar d1, d2;
				d1 = w1->d() + sqrt((Intersect_X - w1->pseudo_x()) * (Intersect_X - w1->pseudo_x()) + (Intersect_Y - w1->pseudo_y()) * (Intersect_Y - w1->pseudo_y()));
				d2 = w2->d() + sqrt((Intersect_X - w2->pseudo_x()) * (Intersect_X - w2->pseudo_x()) + (Intersect_Y - w2->pseudo_y()) * (Intersect_Y - w2->pseudo_y()));

				if (d1 < d2 * NUMERCIAL_EPSILON)
					return w2_invalid;
				if (d2 < d1 * NUMERCIAL_EPSILON)
					return w1_invalid;
			}
		}

		if (w2->start() >= w1->stop())
		{
            Scalar Intersect_X, Intersect_Y;

			// judge the previous order of the two windows
			CalculateIntersectionPoint(w2->pseudo_x(), w2->pseudo_y(), w2->start(), 0, w1->pseudo_x(), w1->pseudo_y(), w1->stop(), 0, Intersect_X, Intersect_Y);

            face_pointer f = opposite_face(Tri.bottom_edge, Tri.face);
            edge_pointer e = next_edge(f, Tri.bottom_edge, Tri.left_vertex);
            Scalar angle = vertex_angle(f, Tri.left_vertex);
            Scalar Cx = this->mesh()->data(e).length * cos(angle);
            Scalar Cy = this->mesh()->data(e).length * -sin(angle);

            if ((PointInTriangle(Intersect_X, Intersect_Y, this->mesh()->data(Tri.bottom_edge).length, Cx, Cy))
				&& (Intersect_Y <= 0) && (Intersect_Y >= w1->pseudo_y()) && (Intersect_Y >= w2->pseudo_y()))
			{
                Scalar d1, d2;
				d1 = w1->d() + sqrt((Intersect_X - w1->pseudo_x()) * (Intersect_X - w1->pseudo_x()) + (Intersect_Y - w1->pseudo_y()) * (Intersect_Y - w1->pseudo_y()));
				d2 = w2->d() + sqrt((Intersect_X - w2->pseudo_x()) * (Intersect_X - w2->pseudo_x()) + (Intersect_Y - w2->pseudo_y()) * (Intersect_Y - w2->pseudo_y()));

				if (d1 < d2 - NUMERCIAL_EPSILON)
					return w2_invalid;
				if (d2 < d1 - NUMERCIAL_EPSILON)
					return w1_invalid;
			}
		}

		return both_valid;
	}

	inline void GeodesicAlgorithmExact::pairwise_windows_checking(list_pointer &list)
	{
		if (list->empty()) return;

		interval_pointer iter = list->begin();
		interval_pointer next, iter_t;

		next = iter->next();

		// traverse successive pairs of windows
		while ((!list->empty()) && (next != NULL))
		{
			windows_state ws = check_between_two_windows(iter, next);

			switch (ws)
			{
			case geodesic::w1_invalid:
				iter_t = iter;
				if (iter == list->begin())
				{
					iter = iter->next();
				}
				else
				{
					iter = iter->previous();
				}

				list->erase(iter_t);
				delete iter_t;
				--m_windows_wavefront;
				break;

			case geodesic::w2_invalid:
				list->erase(next);
				delete next;
				--m_windows_wavefront;
				break;

			case geodesic::both_valid:
				iter = iter->next();
				break;

			default:
				break;
			}

			next = iter->next();
		}
	}

	//------------------------- main operation ----------------------------
	inline void GeodesicAlgorithmExact::propagate_one_windows_list(list_pointer &list)
	{
		if (list->empty()) return;
        OpenMesh::HalfedgeHandle hf0 = this->mesh()->halfedge_handle(list->edge(), hfid0);
        OpenMesh::HalfedgeHandle hf1 = this->mesh()->halfedge_handle(list->edge(), hfid1);
        if (this->mesh()->face_handle(hf0).idx()>-1 && this->mesh()->face_handle(hf1).idx()>-1)
        {
			// Rule 2: pairwise windows checking
            check_with_vertices(list);
			pairwise_windows_checking(list);

            // Rule 1: "One Angle Two Sides"
            find_separating_point(list);
            propagate_windows_to_two_edges(list);
		}
	}

	//-------------------------- main entry --------------------------

	
    inline void GeodesicAlgorithmExact::propagate(unsigned source, std::vector<size_t>& idxs)
	{
		// initialization
        m_source = SubVidxfromMesh[source];
		initialize_propagation_data();
		while (!m_vertex_queue.empty())
		{
			// (1) pop a vertex from M_VERTEX_QUEUE
            vertex_pointer vert = *m_vertex_queue.begin();
			m_vertex_queue.erase(m_vertex_queue.begin());

			// (2) update wavefront
            this->mesh()->data(vert).state = VertexState::INSIDE;
            for(auto e_it = this->mesh()->ve_begin(vert); e_it != this->mesh()->ve_end(vert); ++e_it)
			{
                vertex_pointer vert_it = opposite_vertex(*e_it, vert);
                if (this->mesh()->data(vert_it).state == VertexState::OUTSIDE)
                    this->mesh()->data(vert_it).state = VertexState::FRONT;
			}

			// (3) handle saddle vertex
            if (this->mesh()->data(vert).saddle_or_boundary) create_pseudo_source_windows(vert, false);

            // (4) push window lists on the wavefront incident to v into M_LIST_QUEUE
            for(auto e_it = this->mesh()->ve_begin(vert); e_it != this->mesh()->ve_end(vert); ++e_it)
			{
                edge_pointer edge_it = *e_it;
                if (!interval_list_0(edge_it)->empty())
                {
                    m_list_queue.push(interval_list_0(edge_it));
                }
                if (!interval_list_1(edge_it)->empty())
                {
                    m_list_queue.push(interval_list_1(edge_it));
                }
			}


            for(auto f_it = this->mesh()->vf_begin(vert); f_it != this->mesh()->vf_end(vert); ++f_it)
			{
                edge_pointer   edge_it = opposite_edge(*f_it, vert);
                bool two_adjface = (this->mesh()->face_handle(this->mesh()->halfedge_handle(edge_it, hfid0)).idx()>-1)
                        && (this->mesh()->face_handle(this->mesh()->halfedge_handle(edge_it, hfid1)).idx()>-1);
                vertex_pointer vert_it;
                if(two_adjface)
                {
                    face_pointer faceid = opposite_face(edge_it, *f_it);
                    vert_it = opposite_vertex(faceid, edge_it);
                }
                if (!two_adjface || (this->mesh()->data(vert_it).state != VertexState::OUTSIDE))
                {
                    if (!interval_list_0(edge_it)->empty())
                    {
                        m_list_queue.push(interval_list_0(edge_it));
                    }
                    if (!interval_list_1(edge_it)->empty())
                    {
                        m_list_queue.push(interval_list_1(edge_it));
                    }
                }
			}


			// (5) propagate window lists in a FIFO order
			while (!m_list_queue.empty())
			{
				// pop an list from M_LIST_QUEUE
                list_pointer list = m_list_queue.front();

				m_list_queue.pop();

                bool is_boundary = calculate_triangle_parameters(list, Tri);
				if (!is_boundary)
				{
					// propagate the window list using Rule 1 and 2
					wl_left.clear(); wl_right.clear();
					propagate_one_windows_list(list);

					// merge windows lists
					if (!wl_left.empty())
					{
						// in VTP, both "PrimeMerge" and "SecondMerge" connect window lists in an order-free way
						if (!Tri.left_list->empty())
						{
							Tri.left_list->begin()->previous() = wl_left.end();
							wl_left.end()->next() = Tri.left_list->begin();
                            Tri.left_list->begin() = wl_left.begin();
						}
						else
						{
							Tri.left_list->begin() = wl_left.begin();
                            Tri.left_list->end() = wl_left.end();
						}

						// push updated list into M_LIST_QUEUE
                        vertex_pointer v0 = this->mesh()->from_vertex_handle(this->mesh()->halfedge_handle(Tri.left_edge, hfid0));
                        vertex_pointer v1 = this->mesh()->from_vertex_handle(this->mesh()->halfedge_handle(Tri.left_edge, hfid1));
                        if (((this->mesh()->data(v0).state == VertexState::INSIDE)
                             || (this->mesh()->data(v1).state == VertexState::INSIDE))
                                && (!Tri.left_list->empty()))
                        {
							m_list_queue.push(Tri.left_list);
                        }
					}

					if (!wl_right.empty())
					{
						// in VTP, both "PrimeMerge" and "SecondMerge" connect window lists in an order-free way
						if (!Tri.right_list->empty())
						{
							Tri.right_list->end()->next() = wl_right.begin();
							wl_right.begin()->previous() = Tri.right_list->end();
							Tri.right_list->end() = wl_right.end();
						}
						else
						{
							Tri.right_list->begin() = wl_right.begin();
							Tri.right_list->end() = wl_right.end();
						}

						// push updated list into M_LIST_QUEUE
                        vertex_pointer v0 = this->mesh()->from_vertex_handle(this->mesh()->halfedge_handle(Tri.right_edge, hfid0));
                        vertex_pointer v1 = this->mesh()->from_vertex_handle(this->mesh()->halfedge_handle(Tri.right_edge, hfid1));
                        if (((this->mesh()->data(v0).state == VertexState::INSIDE)
                             || (this->mesh()->data(v1).state == VertexState::INSIDE)) && (!Tri.right_list->empty()))
							m_list_queue.push(Tri.right_list);
					}
				}

				list->clear();
			}
			// statistics
			if (m_vertex_queue.size() > m_queue_max_size)
				m_queue_max_size = m_vertex_queue.size();
		}
        idxs.clear();
        for(auto v_it = this->mesh()->vertices_begin(); v_it != this->mesh()->vertices_end(); ++v_it)
        {
            idxs.push_back(MeshVidxfromSub[v_it->idx()]);
            this->ori_mesh->data(this->ori_mesh->vertex_handle(MeshVidxfromSub[v_it->idx()])).geodesic_distance
                    = this->mesh()->data(*v_it).geodesic_distance;
        }
	}

    // construct sub mesh
    inline bool GeodesicAlgorithmExact::construct_submesh(Mesh* sub_mesh, size_t source_idx, Scalar radius)
    {
        std::queue<size_t> vertexlist;
        vertexlist.push(source_idx);
        Vec3 srcp = ori_mesh->point(ori_mesh->vertex_handle(source_idx));
        std::vector<bool> visited(ori_mesh->n_vertices(), false);
        std::vector<bool> added_face(ori_mesh->n_faces(), false);
        SubVidxfromMesh.resize(ori_mesh->n_vertices());
        SubVidxfromMesh.setConstant(-1);
        MeshVidxfromSub.clear();
        visited[source_idx] = true;
        while(!vertexlist.empty())
        {
            size_t vidx = vertexlist.front();
            vertexlist.pop();
            OpenMesh::VertexHandle vh = ori_mesh->vertex_handle(vidx);
            Vec3 vp = ori_mesh->point(vh);
            if((srcp - vp).norm() < radius)
            {
                vertex_pointer new_v = sub_mesh->add_vertex(vp);
                SubVidxfromMesh[vh.idx()] = new_v.idx();
                MeshVidxfromSub.push_back(vh.idx());
                for(auto vv_it = ori_mesh->vv_begin(vh); vv_it != ori_mesh->vv_end(vh); vv_it++)
                {
                    if(!visited[vv_it->idx()])
                    {
                        vertexlist.push(vv_it->idx());
                        visited[vv_it->idx()] = true;
                    }
                }
                for(auto vf_it = ori_mesh->vf_begin(vh); vf_it != ori_mesh->vf_end(vh); vf_it++)
                {
                    halfedge_handle hf = ori_mesh->halfedge_handle(*vf_it);
                    if(!added_face[vf_it->idx()])
                    {
                        vertex_pointer vh = ori_mesh->from_vertex_handle(hf);
                        vertex_pointer nextv = ori_mesh->to_vertex_handle(hf);
                        vertex_pointer thirdv = ori_mesh->to_vertex_handle(ori_mesh->next_halfedge_handle(hf));
                        if(SubVidxfromMesh[vh.idx()] >= 0
                            && SubVidxfromMesh[nextv.idx()] >= 0
                            && SubVidxfromMesh[thirdv.idx()] >= 0)
                        {
                            std::vector<vertex_pointer> vertices;
                            vertices.push_back(sub_mesh->vertex_handle(SubVidxfromMesh[vh.idx()]));
                            vertices.push_back(sub_mesh->vertex_handle(SubVidxfromMesh[nextv.idx()]));
                            vertices.push_back(sub_mesh->vertex_handle(SubVidxfromMesh[thirdv.idx()]));
                            sub_mesh->add_face(vertices);
                            added_face[vf_it->idx()] = true;
                        }
                    }
                }
            }
        }

        sub_mesh->delete_isolated_vertices();
        sub_mesh->garbage_collection();

        if(sub_mesh->n_vertices() > 0)
            return true;
        else
            return false;
    }

	//---------------------- print statistics --------------------------
	inline void GeodesicAlgorithmExact::print_statistics()
	{
		GeodesicAlgorithmBase::print_statistics();

        Scalar memory = sizeof(Interval);

		//std::cout << std::endl;
		std::cout << "Peak number of intervals on wave-front " << m_windows_peak << std::endl;
		std::cout << "uses about " << memory * m_windows_peak / 1e6 << "MB of memory" << std::endl;
		std::cout << "total interval propagation number " << m_windows_propagation << std::endl;
		std::cout << "maximum interval queue size is " << m_queue_max_size << std::endl;
	}
}		


class Timer
{
public:

	typedef int EventID;

	EventID get_time()
	{
		EventID id = time_values_.size();

#ifdef USE_OPENMP
		time_values_.push_back(omp_get_wtime());
#else
		time_values_.push_back(clock());
#endif

		return id;
	}

	double elapsed_time(EventID event1, EventID event2)
	{
		assert(event1 >= 0 && event1 < static_cast<EventID>(time_values_.size()));
		assert(event2 >= 0 && event2 < static_cast<EventID>(time_values_.size()));

#ifdef USE_OPENMP
		return time_values_[event2] - time_values_[event1];
#else
		return double(time_values_[event2] - time_values_[event1]) / CLOCKS_PER_SEC;
#endif
	}

	void reset()
	{
		time_values_.clear();
	}

private:
#ifdef USE_OPENMP
	std::vector<double> time_values_;
#else
	std::vector<clock_t> time_values_;
#endif
};
namespace svr
{
    //------------------------------------------------------------------------
    //	Define neighborIter class
    //------------------------------------------------------------------------
    class neighborIter
    {
    public:
        neighborIter(const std::map<size_t, Scalar> &nodeNeighbors)
        {
            m_neighborIter = nodeNeighbors.begin();
            m_neighborEnd = nodeNeighbors.end();
        }

        neighborIter& operator++()
        {
            if (m_neighborIter != m_neighborEnd)
                ++m_neighborIter;
            return *this;
        }

        neighborIter operator++(int)
        {
            neighborIter tempIter(*this);
            ++*this;
            return tempIter;
        }

        const std::pair<const size_t, Scalar>& operator*() { return *m_neighborIter; }
        std::map<size_t, Scalar>::const_iterator operator->() { return m_neighborIter; }
        bool is_valid() { return m_neighborIter != m_neighborEnd; }
        size_t getIndex() { return m_neighborIter->first; }
        Scalar getWeight() { return m_neighborIter->second; }

    private:
        std::map<size_t, Scalar>::const_iterator m_neighborIter;
        std::map<size_t, Scalar>::const_iterator m_neighborEnd;
    };

    //------------------------------------------------------------------------
    //	Define node sampling class
    //------------------------------------------------------------------------
    class nodeSampler
    {
    public:
        enum sampleAxis { X_AXIS, Y_AXIS, Z_AXIS };
        nodeSampler() {};

        // return sample radius
        Scalar SampleAndConstuct(Mesh &mesh, Scalar sampleRadiusRatio, const Matrix3X & src_points);
        Scalar SampleAndConstuctAxis(Mesh &mesh, Scalar sampleRadiusRatio, sampleAxis axis);

        Scalar SampleAndConstuctForSrcPoints(Mesh &mesh, Scalar sampleRadiusRatio, const Matrix3X & src_points, const Eigen::MatrixXi & src_knn_indices);
        Scalar SampleAndConstuctFPS(Mesh &mesh, Scalar sampleRadiusRatio, const Matrix3X & src_points, const Eigen::MatrixXi & src_knn_indices, int num_vn, int num_nn);


        void updateWeight(Mesh &mesh);
        void constructGraph(bool is_uniform);

        size_t nodeSize() const { return m_nodeContainer.size(); }

        neighborIter getNodeNodeIter(size_t nodeIdx) const { return neighborIter(m_nodeGraph[nodeIdx]); }
        neighborIter getVertexNodeIter(size_t vertexIdx) const { return neighborIter(m_vertexGraph[vertexIdx]); }
        size_t getNodeVertexIdx(size_t nodeIdx) const { return m_nodeContainer.at(nodeIdx).second; }

        size_t getVertexNeighborSize(size_t vertexIdx) const { return m_vertexGraph.at(vertexIdx).size(); }
        size_t getNodeNeighborSize(size_t nodeIdx) const { return m_nodeGraph.at(nodeIdx).size(); }
        void initWeight(RowMajorSparseMatrix& matPV, VectorX & matP,
			RowMajorSparseMatrix& matB, VectorX& matD, VectorX& smoothw);
        void print_nodes(Mesh & mesh, std::string file_path);

    private:
        size_t m_meshVertexNum = 0;
        size_t m_meshEdgeNum = 0;
        Scalar m_averageEdgeLen = 0.0f;
        Scalar m_sampleRadius = 0.0f;
        std::vector<Scalar> non_unisamples_Radius;
        Mesh * m_mesh;

    public:
        std::vector<std::pair<size_t, size_t>> m_nodeContainer;
        std::vector<std::map<size_t, Scalar>> m_vertexGraph; // vertex node graph
        std::vector<std::map<size_t, Scalar>> m_nodeGraph;
        std::vector<VectorX> m_geoDistContainer;
        Eigen::VectorXi      VertexNodeIdx;
        Eigen::VectorXi      projection_indices;

    };
}
enum CorresType {CLOSEST, LANDMARK};
enum PruningType {SIMPLE, NONE};

struct RegParas
{
    int		max_outer_iters;    // nonrigid max iters
    Scalar	w_smo;              // smoothness weight 
    Scalar	w_rot;               // rotation matrix weight 
	Scalar  w_arap_coarse;      // ARAP weight for coarse alignment
    Scalar  w_arap_fine;        // ARAP weight for fine alignment 
    bool	use_normal_reject;  // use normal reject or not
    bool	use_distance_reject;// use distance reject or not
    Scalar	normal_threshold;
    Scalar	distance_threshold;
    int     rigid_iters;         // rigid registration max iterations
    bool	use_landmark;
    bool    use_fixedvex;        // Set the point which haven't correspondences
    bool    calc_gt_err;         // calculate ground truth error (DEBUG)
    bool    data_use_robust_weight;  // use robust weights for alignment term or not

	bool	use_symm_ppl; // use the symmetric point-to-plane distance or point-to-point distance.

    std::vector<int> landmark_src;
    std::vector<int> landmark_tar;
    std::vector<int> fixed_vertices;

    Scalar  Data_nu;
    Scalar  Data_initk;
    Scalar  Data_endk;
    Scalar  stop_coarse;
    Scalar  stop_fine;

	// Sample para
    Scalar  uni_sample_radio;       // uniform sample radio
    bool    use_geodesic_dist;
    bool    print_each_step_info;   // debug output : each step nodes, correspondences


    // output path
    std::string out_gt_file;
    std::string out_each_step_info;
    int         num_sample_nodes;

    std::vector<Scalar> each_times;
    std::vector<Scalar> each_gt_mean_errs;
    std::vector<Scalar> each_gt_max_errs;
    std::vector<Scalar> each_energys;
    std::vector<Scalar> each_iters;
    std::vector<Vector4> each_term_energy;
    Scalar  non_rigid_init_time;
    Scalar  init_gt_mean_errs;
    Scalar  init_gt_max_errs;

    bool    use_coarse_reg;
    bool    use_fine_reg;

    Scalar  mesh_scale;

    RegParas() // default
    {
        max_outer_iters = 50;
        w_smo = 0.01;  // smooth
        w_rot = 1e-4;   // orth
		w_arap_coarse = 500; // 10;
        w_arap_fine = 200;
        use_normal_reject = false;
        use_distance_reject = false;
        normal_threshold = M_PI / 3;
        distance_threshold = 0.05;
        rigid_iters = 0;
        use_landmark = false;
        use_fixedvex = false;
        calc_gt_err = false;
        data_use_robust_weight = true;

		use_symm_ppl = true;

        Data_nu = 0.0;
        Data_initk = 1;
        Data_endk = 1.0/sqrt(3);
        stop_coarse = 1e-3;
        stop_fine = 1e-4;

        // Sample para
        uni_sample_radio = 5;
        use_geodesic_dist = true;
        print_each_step_info = false;

        non_rigid_init_time = .0;
        init_gt_mean_errs = .0;

        use_coarse_reg = true;
        use_fine_reg = true;
    }

    public:
    void print_params(std::string outf)
    {
        std::ofstream out(outf);
        
    
    out << "setting:\n" <<std::endl;

    // setting
    out << "use_coarse_reg: " << use_coarse_reg << std::endl;
    out << "use_fine_reg: " << use_fine_reg << std::endl;
    out << "use_symm_ppl: " << use_symm_ppl<< std::endl;
    out << "data_use_robust_weight: " << data_use_robust_weight<< std::endl;  // use robust welsch function as energy function or just use L2-norm
    
    out <<"w_smo: "<<w_smo<< std::endl;
    out << "w_rot: "<< w_rot<< std::endl;
	out << "w_arap_coarse: " << w_arap_coarse<< std::endl;
    out << "w_arap_fine: " <<  w_arap_fine<< std::endl;

    out << "uni_sample_radio: " << uni_sample_radio<< std::endl;   
    out << "use_geodesic_dist: " << use_geodesic_dist << std::endl; 
    out << "print_each_step_info: " << print_each_step_info<< std::endl;   
    out << "out_gt_file" << out_gt_file<< std::endl;
    out << "out_each_step_info" << out_each_step_info<< std::endl;

    out << "\n\noutput:\n" <<std::endl;

    // output
    out << "mesh_scale: " << mesh_scale << std::endl;
    out << "num_sample_nodes: " << num_sample_nodes<< std::endl;
    out << "non_rigid_init_time: " << non_rigid_init_time<< std::endl;
    out << "init_gt_mean_errs: " << init_gt_mean_errs<< std::endl;
    out <<"init_gt_max_errs: " << init_gt_max_errs<< std::endl;
    out << "Data_nu: " << Data_nu<< std::endl;

    out << "\n\ndefault:\n" <<std::endl;

    // default
    out << "max_outer_iters: " << max_outer_iters << std::endl;
    out << "stop_coarse: " <<stop_coarse<< std::endl;
    out << "stop_fine: " << stop_fine<< std::endl;
    
    out << "Data_initk: " << Data_initk<< std::endl;
    out << "Data_endk: " << Data_endk<< std::endl;
    out << "calc_gt_err: " << calc_gt_err<< std::endl; 
    out << "use_normal_reject: " <<	use_normal_reject<< std::endl;  // use normal reject or not
    out <<	"use_distance_reject: " << use_distance_reject<< std::endl;
    out << "normal_threshold: " <<	normal_threshold<< std::endl;
    out << "distance_threshold: " <<	distance_threshold<< std::endl;
    
    out << "\n\nuseless:\n" <<std::endl;
    
    // useless 
    out << "use_landmark: " << use_landmark<< std::endl;
    out << "use_fixedvex: " << use_fixedvex<< std::endl;  
    out <<  "rigid_iters: " << rigid_iters << std::endl; 
    out.close();
    }
};

// normalize mesh
Scalar mesh_scaling(Mesh& src_mesh, Mesh& tar_mesh);
// Convert Mesh to libigl format to calculate geodesic distance
void Mesh2VF(Mesh & mesh, MatrixXX& V, Eigen::MatrixXi& F);
Vec3 Eigen2Vec(Vector3 s);
Vector3 Vec2Eigen(Vec3 s);
// read landmark points into landmark_src and landmark_tar if they exist
bool read_landmark(const char* filename, std::vector<int>& landmark_src, std::vector<int>& landmark_tar);
// read fixed points into vertices_list if they exist
bool read_fixedvex(const char* filename, std::vector<int>& vertices_list);

#ifdef __linux__
bool my_mkdir(std::string file_path);
#endif
class Registration
{
public:
    Registration();
    virtual ~Registration();

    Mesh* src_mesh_;
    Mesh* tar_mesh_;
    int n_src_vertex_;
    int n_tar_vertex_;
    int n_landmark_nodes_;
    struct Closest{
        int src_idx; // vertex index from source model
        int tar_idx; // face index from target model
        Vector3 position;
        Vector3 normal;
        Scalar  min_dist2;
    };
    typedef std::vector<Closest> VPairs;

protected:
    // non-rigid Energy function paras
    VectorX weight_d_;	                // robust weight for alignment "\alpha_i"
    RowMajorSparseMatrix mat_A0_;	    // symmetric coefficient matrix for linear equations
    Matrix3X tar_points_;               // target_mesh  (3,m);
    VectorX vec_b_;                     // rights for linear equations
    VectorX corres_U0_;                 // all correspondence points (3,n);

    KDtree* target_tree;                // correspondence paras

    // Rigid paras
    Affine3 rigid_T_;                   // rigid registration transform matrix

    // Check correspondence points
    VectorX corres_pair_ids_;
    VPairs correspondence_pairs_;
    int current_n_;

    // dynamic welsch parasmeters
    bool init_nu;
    Scalar end_nu;
    Scalar nu;



public:
    // adjusted paras
    RegParas pars_;

public:
    virtual Scalar DoNonRigid() { return 0.0; }
    Scalar DoRigid();
    void InitFromInput(Mesh& src_mesh, Mesh& tar_mesh, RegParas& paras);
    virtual void Initialize(){}

private:
    Eigen::VectorXi  init_geo_pairs;

protected:
    //point to point rigid registration
    template <typename Derived1, typename Derived2, typename Derived3>
    Affine3 point_to_point(Eigen::MatrixBase<Derived1>& X,
        Eigen::MatrixBase<Derived2>& Y, const Eigen::MatrixBase<Derived3>& w);

    // Find correct correspondences
    void InitCorrespondence(VPairs & corres);
    void FindClosestPoints(VPairs & corres);
	void FindClosestPoints(VPairs & corres, VectorX & deformed_v);
    void FindClosestPoints(VPairs & corres, VectorX & deformed_v, std::vector<size_t>& sample_indices);

    // Pruning method
    void SimplePruning(VPairs & corres, bool use_distance, bool use_normal);

    // Use landmark;
    void LandMarkCorres(VPairs & correspondence_pairs);

    
    template<typename Derived1>
    Scalar FindKnearestMed(Eigen::MatrixBase<Derived1>& X, int nk);
};


class NonRigidreg : public Registration
{
public:
    NonRigidreg();
    ~NonRigidreg();
    virtual Scalar DoNonRigid();
    virtual void Initialize();

private:
    void welsch_weight(VectorX& r, Scalar p);

    void   CalcNodeRotations();
    Scalar SetMeshPoints(Mesh* mesh, const VectorX & target);

    #ifdef USE_PARDISO
    Eigen::PardisoLDLT<RowMajorSparseMatrix, Eigen::Lower> solver_;
    #else
    Eigen::SimplicialLDLT<RowMajorSparseMatrix, Eigen::Lower> solver_;
    #endif

private:
	// Sample paras
    int		num_sample_nodes;               // (r,) the number of sample nodes
    int     num_graph_edges;
    int     num_edges;

	// simplily nodes storage structure
    svr::nodeSampler src_sample_nodes;

	// variable
    VectorX		            X_;	                        // (12r,) transformations of sample nodes

    // alignment term
	VectorX		            nodes_P_;                   // (3n,) all sample nodes' coordinates
    RowMajorSparseMatrix    align_coeff_PV0_;           // (3n, 12r) coefficient matrix F 
    // smooth term
    RowMajorSparseMatrix	reg_coeff_B_;	            // (6|E_G|, 12r) the smooth between nodes, |E_G| is the edges' number of sample node graph;
    VectorX		            reg_right_D_;	            // (6|E_G|,) the different coordinate between xi and xj
    VectorX			        reg_cwise_weights_;         // (6|E_G|) the smooth weight;
    // rotation matrix term
    VectorX		            nodes_R_;	                // (9r,)  "proj(A)"
    RowMajorSparseMatrix	rigid_coeff_L_;	            // (12r, 12r) "H"
    RowMajorSparseMatrix	rigid_coeff_J_;	            // (9r, 12r)  "Y"
    VectorX		            diff_UP_;                   // aux matrix 
	// ARAP term (coarse)
	VectorX                 arap_laplace_weights_;      // (6E,) 
	Matrix3X                local_rotations_;           // (3n,3) "R"
	RowMajorSparseMatrix    arap_coeff_;                // (6E, 12r) "B"
    RowMajorSparseMatrix    arap_coeff_mul_;            // (12r, 12r) "B^T*B"
	VectorX                 arap_right_;                // (6E,) "L"
    // ARAP term (fine) 
	RowMajorSparseMatrix    arap_coeff_fine_;           // (6E, 3n) "B"
    RowMajorSparseMatrix    arap_coeff_mul_fine_;       // (3n, 3n) "B^T*B"
	VectorX                 arap_right_fine_;           // (6E,)  "Y"

    // point clouds 
	Matrix3X                src_points_;                // (3,n)
	Matrix3X                src_normals_;               // (3,n)
	Matrix3X                deformed_normals_;          // (3,n)
	VectorX                 deformed_points_;           // (3n,)
	Matrix3X                target_normals_;            // (3,n)
	RowMajorSparseMatrix    normals_sum_;               // (n,3n) "N" for alignment term 

    // sampling points & vertices relation matrix
    std::vector<size_t>     sampling_indices_;
    std::vector<int>        vertex_sample_indices_;

    // knn-neighbor indices for source points if no faces. 
    Eigen::MatrixXi         src_knn_indices_;
    // bool                    src_has_faces_; 
    int                     align_sampling_num_ = 3000;
	
    // weights of terms during the optimization process 
    Scalar          w_align;      
    Scalar          w_smo;        
    Scalar          optimize_w_align;  
    Scalar          optimize_w_smo;
    Scalar          optimize_w_rot;
	Scalar			optimize_w_arap;

    void InitWelschParam();
	void FullInARAPCoeff();

	void CalcARAPRight();
	void CalcARAPRightFine();

	void InitRotations();
    void CalcLocalRotations(bool isCoarseAlign);
	void CalcDeformedNormals();
	void InitNormalsSum();
    void CalcNormalsSum();

	void PointwiseFineReg(Scalar nu1);
    void GraphCoarseReg(Scalar nu1);

	Scalar CalcEnergy(Scalar& E_align, Scalar& E_reg,
		Scalar& E_rot, Scalar& E_arap, VectorX & reg_weights);

	Scalar CalcEnergyFine(Scalar& E_align, Scalar& E_arap);
    // Aux_tool function
    Scalar CalcEdgelength(Mesh* mesh, int type);

	
};
template <typename Derived1, typename Derived2, typename Derived3>
Affine3 Registration::point_to_point(Eigen::MatrixBase<Derived1>& X,
    Eigen::MatrixBase<Derived2>& Y,
    const Eigen::MatrixBase<Derived3>& w) {
    /// Normalize weight vector
    VectorX w_normalized = w / w.sum();
    /// De-mean
    Vector3 X_mean, Y_mean;
    for (int i = 0; i<3; ++i) {
        X_mean(i) = (X.row(i).array()*w_normalized.transpose().array()).sum();
        Y_mean(i) = (Y.row(i).array()*w_normalized.transpose().array()).sum();
    }
    X.colwise() -= X_mean;
    Y.colwise() -= Y_mean;

    /// Compute transformation
    Affine3 transformation;
    Matrix33 sigma = X * w_normalized.asDiagonal() * Y.transpose();
    Eigen::JacobiSVD<Matrix33> svd(sigma, Eigen::ComputeFullU | Eigen::ComputeFullV);
    if (svd.matrixU().determinant()*svd.matrixV().determinant() < 0.0) {
        Vector3 S = Vector3::Ones(); S(2) = -1.0;
        transformation.linear().noalias() = svd.matrixV()*S.asDiagonal()*svd.matrixU().transpose();
    }
    else {
        transformation.linear().noalias() = svd.matrixV()*svd.matrixU().transpose();
    }
    transformation.translation().noalias() = Y_mean - transformation.linear()*X_mean;
    /// Re-apply mean
    X.colwise() += X_mean;
    Y.colwise() += Y_mean;
    return transformation;
}
class nonRigid_spare: public REG
{

public:
    // 构造和析构
    nonRigid_spare() = default;
    ~nonRigid_spare() override = default;
    void Reg(const std::string& file_target,
                       const std::string& file_source,
                       const std::string& out_path) override;//配准函数
    
};
#endif

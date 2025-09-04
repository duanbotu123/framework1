#pragma once
#include <string>
#include "spare.h"
#include "median.h"
#include "io_mesh.h"
// #define DEBUG
#if (__cplusplus >= 201402L) || (defined(_MSC_VER) && _MSC_VER >= 1800)
#define MAKE_UNIQUE std::make_unique
#else
#define MAKE_UNIQUE company::make_unique
#endif

#ifdef __linux__		
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#endif




namespace svr
{
    //	Define helper functions
    static auto square = [](const Scalar argu) { return argu * argu; };
    static auto cube = [](const Scalar argu) { return argu * argu * argu; };
    static auto max = [](const Scalar lhs, const Scalar rhs) { return lhs > rhs ? lhs : rhs; };

    //------------------------------------------------------------------------
    //	Node Sampling based on geodesic distance metric
    //
    //	Note that this member function samples nodes along some axis.
    //	Each node is not covered by any other node. And distance between each
    //	pair of nodes is at least sampling radius.
    //------------------------------------------------------------------------
    // Local geodesic calculation
    Scalar nodeSampler::SampleAndConstuctAxis(Mesh &mesh, Scalar sampleRadiusRatio, sampleAxis axis)
    {
        //	Save numbers of vertex and edge
        m_meshVertexNum = mesh.n_vertices();
        m_meshEdgeNum = mesh.n_edges();
        m_mesh = & mesh;

        //	Calculate average edge length of bound mesh
        for (size_t i = 0; i < m_meshEdgeNum; ++i)
        {
            OpenMesh::EdgeHandle eh = mesh.edge_handle(i);
            Scalar edgeLen = mesh.calc_edge_length(eh);
            m_averageEdgeLen += edgeLen;
        }
        m_averageEdgeLen /= m_meshEdgeNum;

        //	Sampling radius is calculated as averageEdgeLen multiplied by sampleRadiusRatio
        m_sampleRadius = sampleRadiusRatio * m_averageEdgeLen;

        //	Reorder mesh vertex along axis
        std::vector<size_t> vertexReorderedAlongAxis(m_meshVertexNum);
        size_t vertexIdx = 0;
        std::generate(vertexReorderedAlongAxis.begin(), vertexReorderedAlongAxis.end(), [&vertexIdx]() -> size_t { return vertexIdx++; });
        std::sort(vertexReorderedAlongAxis.begin(), vertexReorderedAlongAxis.end(), [&mesh, axis](const size_t &lhs, const size_t &rhs) -> bool {
            size_t lhsIdx = lhs;
            size_t rhsIdx = rhs;
            OpenMesh::VertexHandle vhl = mesh.vertex_handle(lhsIdx);
            OpenMesh::VertexHandle vhr = mesh.vertex_handle(rhsIdx);
            Mesh::Point vl = mesh.point(vhl);
            Mesh::Point vr = mesh.point(vhr);
            return vl[axis] > vr[axis];
        });

        //	Sample nodes using radius of m_sampleRadius
        size_t firstVertexIdx = vertexReorderedAlongAxis[0];
        VertexNodeIdx.resize(m_meshVertexNum);
        VertexNodeIdx.setConstant(-1);
        VertexNodeIdx[firstVertexIdx] = 0;
        size_t cur_node_idx = 0;

        m_vertexGraph.resize(m_meshVertexNum);
        VectorX weight_sum = VectorX::Zero(m_meshVertexNum);

        for (auto &vertexIdx : vertexReorderedAlongAxis)
        {
            if(VertexNodeIdx[vertexIdx] < 0 && m_vertexGraph.at(vertexIdx).empty())
            {
                m_nodeContainer.emplace_back(cur_node_idx, vertexIdx);
                VertexNodeIdx[vertexIdx] = cur_node_idx;

                std::vector<size_t> neighbor_verts;
                geodesic::GeodesicAlgorithmExact geoalg(&mesh, vertexIdx, m_sampleRadius);
                geoalg.propagate(vertexIdx, neighbor_verts);
                for(size_t i = 0; i < neighbor_verts.size(); i++)
                {
                    int neighIdx = neighbor_verts[i];
                    Scalar geodist = mesh.data(mesh.vertex_handle(neighIdx)).geodesic_distance;
                    if(geodist < m_sampleRadius)
                    {
                        Scalar weight = std::pow(1-std::pow(geodist/m_sampleRadius, 2), 3);
                        m_vertexGraph.at(neighIdx).emplace(std::pair<int, Scalar>(cur_node_idx, weight));
                        weight_sum[neighIdx] += weight;
                    }
                }
                cur_node_idx++;
            }
        }

        m_nodeGraph.resize(cur_node_idx);
        for (auto &vertexIdx : vertexReorderedAlongAxis)
        {
            for(auto &node: m_vertexGraph[vertexIdx])
            {
                size_t nodeIdx = node.first;
                for(auto &neighNode: m_vertexGraph[vertexIdx])
                {
                    size_t neighNodeIdx = neighNode.first;
                    if(nodeIdx != neighNodeIdx)
                    {
                        m_nodeGraph.at(nodeIdx).emplace(std::pair<int, Scalar>(neighNodeIdx, 1.0));
                    }
                }
                m_vertexGraph.at(vertexIdx).at(nodeIdx) /= weight_sum[vertexIdx];
            }
        }
        return m_sampleRadius;
    }

    //------------------------------------------------------------------------
    //	Node Sampling based on geodesic distance metric
    //
    //	Note that this member function samples nodes along some axis.
    //	Each node is not covered by any other node. And distance between each
    //	pair of nodes is at least sampling radius.
    //------------------------------------------------------------------------
    // Local geodesic calculation
    Scalar nodeSampler::SampleAndConstuct(Mesh &mesh, Scalar sampleRadiusRatio, const Matrix3X & src_points)
    {
        //	Save numbers of vertex and edge
        m_meshVertexNum = mesh.n_vertices();
        m_meshEdgeNum = mesh.n_edges();
        m_mesh = & mesh;
        

        //	Calculate average edge length of bound mesh
        for (size_t i = 0; i < m_meshEdgeNum; ++i)
        {
            OpenMesh::EdgeHandle eh = mesh.edge_handle(i);
            Scalar edgeLen = mesh.calc_edge_length(eh);
            m_averageEdgeLen += edgeLen;
        }
        m_averageEdgeLen /= m_meshEdgeNum;

        //	Sampling radius is calculated as averageEdgeLen multiplied by sampleRadiusRatio
        m_sampleRadius = sampleRadiusRatio * m_averageEdgeLen;
        
        // de-mean
        Vector3 means = src_points.rowwise().mean();
        Matrix3X demean_src_points = src_points.colwise() - means;
        Matrix33 covariance = demean_src_points * demean_src_points.transpose();

        Eigen::SelfAdjointEigenSolver<Matrix33> eigensolver(covariance);
        Vector3 eigen_values = eigensolver.eigenvalues();
        int max_idx = 2;
        if(eigen_values[0] > eigen_values[max_idx])    max_idx = 0;
        if(eigen_values[1] > eigen_values[max_idx])    max_idx = 1;

        Vector3 main_axis = eigensolver.eigenvectors().col(max_idx);

        VectorX projection = demean_src_points.transpose() * main_axis;
        
        projection_indices.resize(projection.size());
        projection_indices.setLinSpaced(projection.size(), 0, projection.size() - 1);

        auto compare = [&projection](int i, int j) {
            return projection(i) < projection(j);
        };

        std::sort(projection_indices.data(), projection_indices.data() + projection_indices.size(), compare);


        //	Sample nodes using radius of m_sampleRadius
        size_t firstVertexIdx = projection_indices[0];
        VertexNodeIdx.resize(m_meshVertexNum);
        VertexNodeIdx.setConstant(-1);
        VertexNodeIdx[firstVertexIdx] = 0;
        size_t cur_node_idx = 0;

        m_vertexGraph.resize(m_meshVertexNum);
        VectorX weight_sum = VectorX::Zero(m_meshVertexNum);

        for(int idx = 0; idx < projection_indices.size(); idx++)
        {
            int vertexIdx = projection_indices[idx];
            if(VertexNodeIdx[vertexIdx] < 0 && m_vertexGraph.at(vertexIdx).empty())
            {
                m_nodeContainer.emplace_back(cur_node_idx, vertexIdx);
                VertexNodeIdx[vertexIdx] = cur_node_idx;

                std::vector<size_t> neighbor_verts;
                geodesic::GeodesicAlgorithmExact geoalg(&mesh, vertexIdx, m_sampleRadius);
                geoalg.propagate(vertexIdx, neighbor_verts);
                for(size_t i = 0; i < neighbor_verts.size(); i++)
                {
                    int neighIdx = neighbor_verts[i];
                    Scalar geodist = mesh.data(mesh.vertex_handle(neighIdx)).geodesic_distance;
                    if(geodist < m_sampleRadius)
                    {
                        Scalar weight = std::pow(1-std::pow(geodist/m_sampleRadius, 2), 3);
                        m_vertexGraph.at(neighIdx).emplace(std::pair<int, Scalar>(cur_node_idx, weight));
                        weight_sum[neighIdx] += weight;
                    }
                }
                cur_node_idx++;
            }
        }

        m_nodeGraph.resize(cur_node_idx);
        for(int idx = 0; idx < projection_indices.size(); idx++)
        {
            int vertexIdx = projection_indices[idx];
            for(auto &node: m_vertexGraph[vertexIdx])
            {
                size_t nodeIdx = node.first;
                for(auto &neighNode: m_vertexGraph[vertexIdx])
                {
                    size_t neighNodeIdx = neighNode.first;
                    if(nodeIdx != neighNodeIdx)
                    {
                        m_nodeGraph.at(nodeIdx).emplace(std::pair<int, Scalar>(neighNodeIdx, 1.0));
                    }
                }
                m_vertexGraph.at(vertexIdx).at(nodeIdx) /= weight_sum[vertexIdx];
            }
        }

        

        // // check neighbors 
        // for(int nodeIdx = 0; nodeIdx < cur_node_idx; nodeIdx++)
        // {
        //     int num_neighbors =  m_nodeGraph[nodeIdx].size();
        //     // for (auto &eachNeighbor : m_nodeGraph[nodeIdx])
        //     // {
        //     //     num_neighbors++;
        //     // }
        //     if(num_neighbors<2)
        //     {
                
        //         VectorX dists(cur_node_idx);
        //         #pragma omp parallel for
        //         for(int ni = 0; ni < cur_node_idx; ni++)
        //         {
        //             int vidx0 = getNodeVertexIdx(nodeIdx);
        //             int vidx1 = getNodeVertexIdx(ni);
        //             Scalar dist = (src_points.col(vidx0) - src_points.col(vidx1)).squaredNorm();
        //             dists[ni] = dist;
        //         }
        //         dists[nodeIdx] = 1e10;
        //         for(int k = 0; k < 6; k++)
        //         {
        //             int min_idx = -1;
        //             dists.minCoeff(&min_idx);
        //             m_nodeGraph.at(nodeIdx).emplace(std::pair<int, Scalar>(min_idx, 1.0));
        //             dists[min_idx] = 1e10;
        //         }
        //     }
        // }
        // Eigen::VectorXi vnn(cur_node_idx);
        // // check vertex neighbors 
        // for(int vidx = 0; vidx < cur_node_idx; vidx++)
        // {
        //     // std::cout << "vidx.neighbor_node = " << m_vertexGraph.at(vidx).size() << std::endl;
        //     vnn[vidx] = m_nodeGraph.at(vidx).size();
        // }
        // std::cout << "v neighbor max = " << vnn.maxCoeff() << " min = " << vnn.minCoeff() << std::endl;
        return m_sampleRadius;
    }


    Scalar nodeSampler::SampleAndConstuctForSrcPoints(Mesh &mesh, Scalar sampleRadiusRatio, const Matrix3X & src_points, const Eigen::MatrixXi & src_knn_indices)
    {
        //	Save numbers of vertex and edge
        m_meshVertexNum = src_points.cols();
        int knn_num_neighbor = src_knn_indices.rows(); 
        m_meshEdgeNum = m_meshVertexNum*knn_num_neighbor;
        m_mesh = & mesh;

        //	Calculate average edge length of bound mesh
        for (size_t i = 0; i < m_meshVertexNum; ++i)
        {
            for(size_t j = 0; j < knn_num_neighbor; j++)
            {
                Scalar edgeLen = (src_points.col(i) - src_points.col(src_knn_indices(j, i))).norm();
                m_averageEdgeLen += edgeLen;
            }
        }
        m_averageEdgeLen /= m_meshEdgeNum;


        //	Sampling radius is calculated as averageEdgeLen multiplied by sampleRadiusRatio
        m_sampleRadius = sampleRadiusRatio * m_averageEdgeLen;

        // de-mean
        Vector3 means = src_points.rowwise().mean();
        Matrix3X demean_src_points = src_points.colwise() - means;
        Matrix33 covariance = demean_src_points * demean_src_points.transpose();

        Eigen::SelfAdjointEigenSolver<Matrix33> eigensolver(covariance);
        Vector3 eigen_values = eigensolver.eigenvalues();
        int max_idx = 2;
        if(eigen_values[0] > eigen_values[max_idx])    max_idx = 0;
        if(eigen_values[1] > eigen_values[max_idx])    max_idx = 1;

        Vector3 main_axis = eigensolver.eigenvectors().col(max_idx);

        VectorX projection = demean_src_points.transpose() * main_axis;
        
        projection_indices.resize(projection.size());
        projection_indices.setLinSpaced(projection.size(), 0, projection.size() - 1);

        auto compare = [&projection](int i, int j) {
            return projection(i) < projection(j);
        };

        std::sort(projection_indices.data(), projection_indices.data() + projection_indices.size(), compare);


        //	Sample nodes using radius of m_sampleRadius
        size_t firstVertexIdx = projection_indices[0];
        VertexNodeIdx.resize(m_meshVertexNum);
        VertexNodeIdx.setConstant(-1);
        VertexNodeIdx[firstVertexIdx] = 0;
        size_t cur_node_idx = 0;

        m_vertexGraph.resize(m_meshVertexNum);
        VectorX weight_sum = VectorX::Zero(m_meshVertexNum);

        for(int idx = 0; idx < projection_indices.size(); idx++)
        {
            int vertexIdx = projection_indices[idx];
            if(VertexNodeIdx[vertexIdx] < 0 && m_vertexGraph.at(vertexIdx).empty())
            {
                m_nodeContainer.emplace_back(cur_node_idx, vertexIdx);
                VertexNodeIdx[vertexIdx] = cur_node_idx;
                
                std::queue<size_t> neighbor_verts;
                neighbor_verts.push(vertexIdx);
                Eigen::VectorXi visited(m_meshVertexNum);
                visited.setZero();
                visited[vertexIdx] = 1;
                while(!neighbor_verts.empty())
                {
                    size_t vidx = neighbor_verts.front();
                    neighbor_verts.pop();
                    Scalar dist = (src_points.col(vertexIdx) - src_points.col(vidx)).norm();

                    if(dist < m_sampleRadius)
                    {
                        Scalar weight = std::pow(1-std::pow(dist/m_sampleRadius, 2), 3);
                        m_vertexGraph.at(vidx).emplace(std::pair<int, Scalar>(cur_node_idx, weight));
                        weight_sum[vidx] += weight;
                        for(int j = 0; j < knn_num_neighbor; j++)
                        {
                            int neighbor_vidx = src_knn_indices(j, vidx);
                            if(visited[neighbor_vidx]==0)
                            {
                                neighbor_verts.push(neighbor_vidx);
                                visited[neighbor_vidx]= 1;
                            }
                        }
                    }      
                }

                cur_node_idx++;             
            }
        }

        m_nodeGraph.resize(cur_node_idx);
        for(int idx = 0; idx < projection_indices.size(); idx++)
        {
            int vertexIdx = projection_indices[idx];
            for(auto &node: m_vertexGraph[vertexIdx])
            {
                size_t nodeIdx = node.first;
                for(auto &neighNode: m_vertexGraph[vertexIdx])
                {
                    size_t neighNodeIdx = neighNode.first;
                    if(nodeIdx != neighNodeIdx)
                    {
                        m_nodeGraph.at(nodeIdx).emplace(std::pair<int, Scalar>(neighNodeIdx, 1.0));
                    }
                }
                m_vertexGraph.at(vertexIdx).at(nodeIdx) /= weight_sum[vertexIdx];
            }
        }

        // // check neighbors 
        // for(int nodeIdx = 0; nodeIdx < cur_node_idx; nodeIdx++)
        // {
        //     int num_neighbors = 0;
        //     for (auto &eachNeighbor : m_nodeGraph[nodeIdx])
        //     {
        //         num_neighbors++;
        //     }
        //     if(num_neighbors==0)
        //     {
                
        //         VectorX dists(cur_node_idx);
        //         #pragma omp parallel for
        //         for(int ni = 0; ni < cur_node_idx; ni++)
        //         {
        //             int vidx0 = getNodeVertexIdx(nodeIdx);
        //             int vidx1 = getNodeNeighborSize(ni);
        //             Scalar dist = (src_points.col(vidx0) - src_points.col(vidx1)).squaredNorm();
        //             dists[ni] = dist;
        //         }
        //         dists[nodeIdx] = 1e10;
        //         int min_idx = -1;
        //         dists.minCoeff(&min_idx);
        //         m_nodeGraph.at(nodeIdx).emplace(std::pair<int, Scalar>(min_idx, 1.0));
        //     }
        // }
        return m_sampleRadius;
    }


    Scalar nodeSampler::SampleAndConstuctFPS(Mesh &mesh, Scalar sampleRadiusRatio, const Matrix3X & src_points, const Eigen::MatrixXi & src_knn_indices, int num_vn, int num_nn)
    {
        m_meshVertexNum = src_points.cols();
        int knn_num_neighbor = src_knn_indices.rows(); 
        m_meshEdgeNum = m_meshVertexNum*knn_num_neighbor;
        m_mesh = & mesh;

        //	Calculate average edge length of bound mesh
        for (size_t i = 0; i < m_meshVertexNum; ++i)
        {
            for(size_t j = 0; j < knn_num_neighbor; j++)
            {
                Scalar edgeLen = (src_points.col(i) - src_points.col(src_knn_indices(j, i))).norm();
                m_averageEdgeLen += edgeLen;
            }
        }
        m_averageEdgeLen /= m_meshEdgeNum;

        //	Sampling radius is calculated as averageEdgeLen multiplied by sampleRadiusRatio
        m_sampleRadius = sampleRadiusRatio * m_averageEdgeLen;

        // start points
        size_t startIndex = 0;
        // FPS to get sampling points in align term
        VectorX minDistances(m_meshVertexNum);
        minDistances.setConstant(std::numeric_limits<Scalar>::max());
        minDistances[startIndex] = 0;
        m_nodeContainer.emplace_back(startIndex, 0);
    
        Scalar minimal_dist = 1e10;
        int cur_node_idx = 1;
        // repeat select farthest points
        while (minimal_dist > m_sampleRadius) {
            // calculate the distance between each point with the sampling points set.         
            #pragma omp parallel for
            for (size_t i = 0; i < m_meshVertexNum; ++i) {
                if(i==startIndex)
                    continue;
                
                Scalar dist = (src_points.col(startIndex) - src_points.col(i)).norm();
                if(dist < minDistances[i])
                    minDistances[i] = dist;
            }

            // choose farthest point
            int maxDistanceIndex;
            minimal_dist = minDistances.maxCoeff(&maxDistanceIndex);
            minDistances[maxDistanceIndex] = 0;

            // add the farthest point into the sampling points set.
            startIndex = maxDistanceIndex;
            m_nodeContainer.emplace_back(cur_node_idx, startIndex);
            cur_node_idx++;
        }

        Matrix3X node_positions(3, cur_node_idx);
        for(int i = 0; i < cur_node_idx; i++)
        {
            int vidx = getNodeVertexIdx(i);
            node_positions.col(i) = src_points.col(vidx);
        }
        KDtree node_tree(node_positions);

        // For each vertex, find num_vn-closest nodes
        m_vertexGraph.resize(m_meshVertexNum);
        VectorX weight_sum(m_meshVertexNum);
        weight_sum.setZero();
        #pragma omp parallel for 
        for(int vidx = 0; vidx < m_meshVertexNum; vidx++)
        {
            std::vector<int> out_indices(num_vn);
            std::vector<Scalar> out_dists(num_vn);
            node_tree.query(src_points.col(vidx).data(), num_vn, out_indices.data(), out_dists.data());
            for(int k = 0; k < num_vn; k++)
            {
                Scalar weight = std::pow(1-std::pow(out_dists[k]/m_sampleRadius, 2), 3);
                m_vertexGraph.at(vidx).emplace(std::pair<int, Scalar>(out_indices[k], weight));
                weight_sum[vidx] += weight;
            }
        }

        #pragma omp parallel for 
        for(int vidx = 0; vidx < m_meshVertexNum; vidx++)
        {
            for(auto &neighNode: m_vertexGraph[vidx])
            {
                size_t neighNodeIdx = neighNode.first;
                m_vertexGraph.at(vidx).at(neighNodeIdx) /= weight_sum[vidx];
            }
        }


        // For each node, find num_nn-closest nodes
        m_nodeGraph.resize(cur_node_idx);
        #pragma omp parallel for 
        for(int nidx = 0; nidx < cur_node_idx; nidx++)
        {
            std::vector<int> out_indices(num_nn+1);
            std::vector<Scalar> out_dists(num_nn+1);
            int vidx = getNodeVertexIdx(nidx);
            node_tree.query(src_points.col(vidx).data(), num_nn+1, out_indices.data(), out_dists.data());
            for(int k = 0; k < num_nn; k++)
            {
                m_nodeGraph.at(nidx).emplace(std::pair<int, Scalar>(out_indices[k+1], 1.0));
            }
        }        
        return m_sampleRadius;
    }


    


    void nodeSampler::initWeight(RowMajorSparseMatrix& matPV, VectorX & matP, RowMajorSparseMatrix& matB, VectorX& matD, VectorX& smoothw)
    {
        Timer time;
        std::vector<Eigen::Triplet<Scalar>> coeff;
        matP.setZero();
        Eigen::VectorXi nonzero_num = Eigen::VectorXi::Zero(m_mesh->n_vertices());
        // data coeff
        for (size_t vertexIdx = 0; vertexIdx < m_meshVertexNum; ++vertexIdx)
        {
            Mesh::Point vi = m_mesh->point(m_mesh->vertex_handle(vertexIdx));
            for (auto &eachNeighbor : m_vertexGraph[vertexIdx])
            {
                size_t nodeIdx = eachNeighbor.first;
                Scalar weight = m_vertexGraph.at(vertexIdx).at(nodeIdx);
                Mesh::Point pj = m_mesh->point(m_mesh->vertex_handle(getNodeVertexIdx(nodeIdx)));

				for (int k = 0; k < 3; k++)
				{
					coeff.push_back(Eigen::Triplet<Scalar>(3 * vertexIdx + k, nodeIdx * 12 + k, weight * (vi[0] - pj[0])));
					coeff.push_back(Eigen::Triplet<Scalar>(3 * vertexIdx + k, nodeIdx * 12 + k + 3, weight * (vi[1] - pj[1])));
					coeff.push_back(Eigen::Triplet<Scalar>(3 * vertexIdx + k, nodeIdx * 12 + k + 6, weight * (vi[2] - pj[2])));
					coeff.push_back(Eigen::Triplet<Scalar>(3 * vertexIdx + k, nodeIdx * 12 + k + 9, weight * 1.0));
				}
                
				matP[vertexIdx * 3] += weight * pj[0];
				matP[vertexIdx * 3 + 1] += weight * pj[1];
				matP[vertexIdx * 3 + 2] += weight * pj[2];
            }
            nonzero_num[vertexIdx] = m_vertexGraph[vertexIdx].size();
        }
        matPV.setFromTriplets(coeff.begin(), coeff.end());


        // smooth coeff
        coeff.clear();
        int max_edge_num = nodeSize() * (nodeSize()-1);
		matB.resize(max_edge_num * 3, 12 * nodeSize());
		matD.resize(max_edge_num * 3);
        smoothw.resize(max_edge_num*3);
		smoothw.setZero();
        matD.setZero();
        int edge_id = 0;

        for (size_t nodeIdx = 0; nodeIdx < m_nodeContainer.size(); ++nodeIdx)
        {
            size_t vIdx0 = getNodeVertexIdx(nodeIdx);
            Mesh::VertexHandle vh0 = m_mesh->vertex_handle(vIdx0);
            Mesh::Point v0 = m_mesh->point(vh0);

            for (auto &eachNeighbor : m_nodeGraph[nodeIdx])
            {
                size_t neighborIdx = eachNeighbor.first;
                size_t vIdx1 = getNodeVertexIdx(neighborIdx);
                Mesh::Point v1 = m_mesh->point(m_mesh->vertex_handle(vIdx1));
                Mesh::Point dv = v0 - v1;
                int k = edge_id;

				for (int t = 0; t < 3; t++)
				{
					coeff.push_back(Eigen::Triplet<Scalar>(k * 3 + t, neighborIdx * 12 + t, dv[0]));
					coeff.push_back(Eigen::Triplet<Scalar>(k * 3 + t, neighborIdx * 12 + t + 3, dv[1]));
					coeff.push_back(Eigen::Triplet<Scalar>(k * 3 + t, neighborIdx * 12 + t + 6, dv[2]));
					coeff.push_back(Eigen::Triplet<Scalar>(k * 3 + t, neighborIdx * 12 + t + 9, 1.0));
					coeff.push_back(Eigen::Triplet<Scalar>(k * 3 + t, nodeIdx * 12 + t + 9, -1.0));
				}

                Scalar dist = dv.norm();
                if(dist > 0)
                {
					smoothw[k * 3] = smoothw[k * 3 + 1] = smoothw[k * 3 + 2] = 1.0 / dist;
                }
                else
                {
					//smoothw[k * 3] = 0.0;
                    std::cout << "node repeat";
                    exit(1);
                }
				matD[k * 3] = dv[0];
				matD[k * 3 + 1] = dv[1];
				matD[k * 3 + 2] = dv[2];
                edge_id++;
            }
        }
        matB.setFromTriplets(coeff.begin(), coeff.end());
        matD.conservativeResize(edge_id*3);
        matB.conservativeResize(edge_id*3, matPV.cols());
        smoothw.conservativeResize(edge_id*3);
        smoothw *= edge_id/(smoothw.sum()/3.0);
    }


    void nodeSampler::print_nodes(Mesh & mesh, std::string file_path)
    {
        std::string namev = file_path + "nodes.obj";
        std::ofstream out1(namev);
        for (size_t i = 0; i < m_nodeContainer.size(); i++)
        {
            int vexid = m_nodeContainer[i].second;
            out1 << "v " << mesh.point(mesh.vertex_handle(vexid))[0] << " " << mesh.point(mesh.vertex_handle(vexid))[1]
                << " " << mesh.point(mesh.vertex_handle(vexid))[2] << std::endl;
        }
        Eigen::VectorXi nonzero_num = Eigen::VectorXi::Zero(m_nodeContainer.size());
        for (size_t nodeIdx = 0; nodeIdx < m_nodeContainer.size(); ++nodeIdx)
        {
            for (auto &eachNeighbor : m_nodeGraph[nodeIdx])
            {
                size_t neighborIdx = eachNeighbor.first;
                out1 << "l " << nodeIdx+1 << " " << neighborIdx+1 << std::endl;
            }
            nonzero_num[nodeIdx] = m_nodeGraph[nodeIdx].size();
        }
        std::cout << "node neighbor min = " << nonzero_num.minCoeff() << " max = "
                  << nonzero_num.maxCoeff() << " average = " << nonzero_num.mean() << std::endl;
        out1.close();
    }
}

Vec3 Eigen2Vec(Vector3 s)
{
    return Vec3(s[0], s[1], s[2]);
}

Vector3 Vec2Eigen(Vec3 s)
{
    return Vector3(s[0], s[1], s[2]);
}

Scalar mesh_scaling(Mesh& src_mesh, Mesh& tar_mesh)
{
    Vec3 max(-1e12, -1e12, -1e12);
    Vec3 min(1e12, 1e12, 1e12);
    for(auto it = src_mesh.vertices_begin(); it != src_mesh.vertices_end(); it++)
    {
        for(int j = 0; j < 3; j++)
        {
            if(src_mesh.point(*it)[j] > max[j])
            {
                max[j] = src_mesh.point(*it)[j];
            }
            if(src_mesh.point(*it)[j] < min[j])
            {
                min[j] = src_mesh.point(*it)[j];
            }
        }
    }

    for(auto it = tar_mesh.vertices_begin(); it != tar_mesh.vertices_end(); it++)
    {
        for(int j = 0; j < 3; j++)
        {
            if(tar_mesh.point(*it)[j] > max[j])
            {
                max[j] = tar_mesh.point(*it)[j];
            }
            if(tar_mesh.point(*it)[j] < min[j])
            {
                min[j] = tar_mesh.point(*it)[j];
            }
        }
    }
    Scalar scale = (max-min).norm();

    for(auto it = src_mesh.vertices_begin(); it != src_mesh.vertices_end(); it++)
    {
        Vec3 p = src_mesh.point(*it);
        p = p/scale;
        src_mesh.set_point(*it, p);
    }

    for(auto it = tar_mesh.vertices_begin(); it != tar_mesh.vertices_end(); it++)
    {
        Vec3 p = tar_mesh.point(*it);
        p = p/scale;
        tar_mesh.set_point(*it, p);
    }

    return scale;
}

void Mesh2VF(Mesh & mesh, MatrixXX& V, Eigen::MatrixXi& F)
{
    V.resize(mesh.n_vertices(),3);
    F.resize(mesh.n_faces(),3);
    for (auto it = mesh.vertices_begin(); it != mesh.vertices_end(); it++)
    {
        V(it->idx(), 0) = mesh.point(*it)[0];
        V(it->idx(), 1) = mesh.point(*it)[1];
        V(it->idx(), 2) = mesh.point(*it)[2];
    }

    for (auto fit = mesh.faces_begin(); fit != mesh.faces_end(); fit++)
    {
        int i = 0;
        for (auto vit = mesh.fv_begin(*fit); vit != mesh.fv_end(*fit); vit++)
        {
            F(fit->idx(), i) = vit->idx();
            i++;
            if (i > 3)
            {
                std::cout << "Error!! one face has more than 3 points!" << std::endl;
                break;
            }
        }
    }
}

bool read_landmark(const char* filename, std::vector<int>& landmark_src, std::vector<int>& landmark_tar)
{
    std::ifstream in(filename);
    std::cout << "filename = " << filename << std::endl;
    if (!in)
    {
        std::cout << "Can't open the landmark file!!" << std::endl;
        return false;
    }
    int x, y;
    landmark_src.clear();
    landmark_tar.clear();
    while (!in.eof())
    {
        if (in >> x >> y) {
            landmark_src.push_back(x);
            landmark_tar.push_back(y);
        }
    }
    in.close();
    std::cout << "landmark_src = " << landmark_src.size() << " tar = " << landmark_tar.size() << std::endl;
    return true;
}

bool read_fixedvex(const char* filename, std::vector<int>& vertices_list)
{
    std::ifstream in(filename);
    std::cout << "filename = " << filename << std::endl;
    if (!in)
    {
        std::cout << "Can't open the landmark file!!" << std::endl;
        return false;
    }
    int x;
    vertices_list.clear();
    while (!in.eof())
    {
        if (in >> x) {
            vertices_list.push_back(x);
        }
    }
    in.close();
    std::cout << "the number of fixed vertices = " << vertices_list.size() << std::endl;
    return true;
}

#ifdef __linux__
bool my_mkdir(std::string file_path)
{
    if(access(file_path.c_str(), 06))
   {
       std::cout << "file_path : (" << file_path << ") didn't exist or no write ability!!" << std::endl;
       if(mkdir(file_path.c_str(), S_IRWXU))
       {
           std::cout << "mkdir " << file_path << " is wrong! please check upper path " << std::endl;
           exit(0);
       }
       std::cout<< "mkdir " << file_path << " success!! " << std::endl;
   }
}
#endif

Registration::Registration() {
    target_tree = NULL;
    src_mesh_ = NULL;
    tar_mesh_ = NULL;
};

Registration::~Registration()
{
    if (target_tree != NULL)
    {
        delete target_tree;
        target_tree = NULL;
    }
}

// initialize from input
void Registration::InitFromInput(Mesh & src_mesh, Mesh & tar_mesh, RegParas& paras)
{
    src_mesh_ = new Mesh;
    tar_mesh_ = new Mesh;
    src_mesh_ = &src_mesh;
    tar_mesh_ = &tar_mesh;
    pars_ = paras;
    n_src_vertex_ = src_mesh_->n_vertices();
    n_tar_vertex_ = tar_mesh_->n_vertices();
    corres_pair_ids_.resize(n_src_vertex_);
    tar_points_.resize(3, n_tar_vertex_);

    for (int i = 0; i < n_tar_vertex_; i++)
    {
        tar_points_(0, i) = tar_mesh_->point(tar_mesh_->vertex_handle(i))[0];
        tar_points_(1, i) = tar_mesh_->point(tar_mesh_->vertex_handle(i))[1];
        tar_points_(2, i) = tar_mesh_->point(tar_mesh_->vertex_handle(i))[2];
    }

        // construct kd Tree
    target_tree = new KDtree(tar_points_);
}



// Rigid Registration
Scalar Registration::DoRigid()
{
    Matrix3X rig_tar_v = Matrix3X::Zero(3, n_src_vertex_);
    Matrix3X rig_src_v = Matrix3X::Zero(3, n_src_vertex_);
    Affine3 old_T;
    InitCorrespondence(correspondence_pairs_);
	corres_pair_ids_.setZero();
    for(int iter = 0; iter < pars_.rigid_iters; iter++)
    {
        for (size_t i = 0; i < correspondence_pairs_.size(); i++)
        {
            rig_src_v.col(i) = Eigen::Map<Vector3>(src_mesh_->point(src_mesh_->vertex_handle(correspondence_pairs_[i].src_idx)).data(), 3, 1);
            rig_tar_v.col(i) = correspondence_pairs_[i].position;
            corres_pair_ids_[correspondence_pairs_[i].src_idx] = 1;
        }
        old_T = rigid_T_;
        rigid_T_ = point_to_point(rig_src_v, rig_tar_v, corres_pair_ids_);

        if((old_T.matrix() - rigid_T_.matrix()).norm() < 1e-3)
        {
            break;
        }

        #pragma omp parallel for
        for (int i = 0; i < n_src_vertex_; i++)
        {
            Vec3 p = src_mesh_->point(src_mesh_->vertex_handle(i));
            Vector3 temp = rigid_T_ * Eigen::Map<Vector3>(p.data(), 3);
            p[0] = temp[0];
            p[1] = temp[1];
            p[2] = temp[2];
            src_mesh_->set_point(src_mesh_->vertex_handle(i), p);
        }

        // Find correspondence
        FindClosestPoints(correspondence_pairs_);
        SimplePruning(correspondence_pairs_, pars_.use_distance_reject, pars_.use_normal_reject);
    }

    return 0;
}

void Registration::FindClosestPoints(VPairs & corres)
{
    corres.resize(n_src_vertex_);

    #pragma omp parallel for
    for (int i = 0; i < n_src_vertex_; i++)
    {
        Scalar mini_dist2;
        int idx = target_tree->closest(src_mesh_->point(src_mesh_->vertex_handle(i)).data(), mini_dist2);
        Closest c;
        c.src_idx = i;
        c.position = tar_points_.col(idx);
        c.normal = Vec2Eigen(tar_mesh_->normal(tar_mesh_->vertex_handle(idx)));
        c.min_dist2 = mini_dist2;
        c.tar_idx = idx;
        corres[i] = c;
    }
}

void Registration::FindClosestPoints(VPairs & corres, VectorX & deformed_v)
{
	corres.resize(n_src_vertex_);
#pragma omp parallel for
	for (int i = 0; i < n_src_vertex_; i++)
	{
		Scalar mini_dist2;
		int idx = target_tree->closest(deformed_v.data() + 3*i, mini_dist2);
		Closest c;
		c.src_idx = i;
		c.position = tar_points_.col(idx);
		c.min_dist2 = mini_dist2;
		c.tar_idx = idx;
		corres[i] = c;
	}
}

void Registration::FindClosestPoints(VPairs & corres, VectorX & deformed_v, std::vector<size_t>& sample_indices)
{
    corres.resize(sample_indices.size());
#pragma omp parallel for
    for(int i = 0; i < sample_indices.size(); i++)
    {
        int sidx = sample_indices[i];
        Scalar mini_dist2;
        int tidx = target_tree->closest(deformed_v.data() + 3*sidx, mini_dist2);
            Closest c;
        c.src_idx = sidx;
        c.position = tar_points_.col(tidx);
        c.min_dist2 = mini_dist2;
        c.tar_idx = tidx;
        corres[i] = c;
    }
}


void Registration::SimplePruning(VPairs & corres, bool use_distance = true, bool use_normal = true)
{
    // Distance and normal
    VectorX tar_min_dists(n_tar_vertex_);
    tar_min_dists.setConstant(1e10);
    Eigen::VectorXi min_idxs(n_tar_vertex_);
    min_idxs.setConstant(-1);

    VectorX corres_idx = VectorX::Zero(n_src_vertex_);
    for(size_t i = 0; i < corres.size(); i++)
    {
        Vector3 closet = corres[i].position;
        Scalar dist = (src_mesh_->point(src_mesh_->vertex_handle(corres[i].src_idx))
                       - Eigen2Vec(closet)).norm();
        Vec3 src_normal = src_mesh_->normal(src_mesh_->vertex_handle(corres[i].src_idx));
        Vec3 tar_normal = Eigen2Vec(corres[i].normal);

        Scalar angle = acos(src_normal | tar_normal / (src_normal.norm()*tar_normal.norm()));
        if((!use_distance || dist < pars_.distance_threshold)
            && (!use_normal || src_mesh_->n_faces() == 0 || angle < pars_.normal_threshold))
        {

            corres_idx[i] = 1;
        }
    }
    if(pars_.use_fixedvex)
    {
        for(size_t i = 0; i < pars_.fixed_vertices.size(); i++)
        {
            int idx = pars_.fixed_vertices[i];
            corres_idx[idx] = 0;
        }
    }
    VPairs corres2;
    for (auto it = corres.begin(); it != corres.end(); it++)
    {
        if (corres_idx[(*it).src_idx] == 1)
        {
            corres2.push_back(*it);
        }
    }
    corres.clear();
    corres = corres2;
}


void Registration::LandMarkCorres(VPairs & corres)
{
    corres.clear();
    if (pars_.landmark_src.size() != pars_.landmark_tar.size())
    {
        std::cout << "Error: landmark data wrong!!" << std::endl;
    }
    n_landmark_nodes_ = pars_.landmark_tar.size();
    for (int i = 0; i < n_landmark_nodes_; i++)
    {
        Closest c;
        c.src_idx = pars_.landmark_src[i];
        OpenMesh::VertexHandle vh = tar_mesh_->vertex_handle(pars_.landmark_tar[i]);

        if (c.src_idx > n_src_vertex_ || c.src_idx < 0)
            std::cout << "Error: source index in Landmark is out of range!" << std::endl;
        if (vh.idx() < 0)
            std::cout << "Error: target index in Landmark is out of range!" << std::endl;

        c.position = Vec2Eigen(tar_mesh_->point(vh));
        c.normal = Vec2Eigen(tar_mesh_->normal(vh));
        corres.push_back(c);
	}
    std::cout << " use landmark and landmark is ... " << pars_.landmark_src.size() << std::endl;
}

void Registration::InitCorrespondence(VPairs & corres)
{
    if(pars_.use_landmark)
    {
        corres.clear();
        for(size_t i = 0; i < pars_.landmark_src.size(); i++)
        {
            Closest c;
            c.src_idx = pars_.landmark_src[i];
            c.tar_idx = pars_.landmark_tar[i];
            c.position = tar_points_.col(c.tar_idx);
            c.normal = Vec2Eigen(tar_mesh_->normal(tar_mesh_->vertex_handle(c.tar_idx)));
            corres.push_back(c);
        }
    }
    else
    {
        FindClosestPoints(corres);
    }
}



/// @param Source (one 3D point per column)
/// @param Target (one 3D point per column)
/// @param Confidence weights


NonRigidreg::NonRigidreg() {
};

NonRigidreg::~NonRigidreg()
{
}

void NonRigidreg::Initialize()
{
	Timer timer;
	
    InitWelschParam();
	
	src_points_.resize(3, n_src_vertex_);
	src_normals_.resize(3, n_src_vertex_);
	corres_U0_.resize(3* n_src_vertex_);

	#pragma omp parallel for
	for (int i = 0; i < n_src_vertex_; i++)
	{
		Vec3 p = src_mesh_->point(src_mesh_->vertex_handle(i));
		src_points_(0, i) = p[0];
		src_points_(1, i) = p[1];
		src_points_(2, i) = p[2];
		Vec3 n = src_mesh_->normal(src_mesh_->vertex_handle(i));
		src_normals_(0, i) = n[0];
		src_normals_(1, i) = n[1];
		src_normals_(2, i) = n[2];
	}
	deformed_normals_ = src_normals_;
	deformed_points_ = Eigen::Map<VectorX>(src_points_.data(), 3*n_src_vertex_);

	int knn_num_neighbor = 6;
	if(!pars_.use_geodesic_dist)
	{
		src_knn_indices_.resize(knn_num_neighbor, n_src_vertex_);
		KDtree* src_tree = new KDtree(src_points_);
		#pragma omp parallel for
		for(int i = 0; i < n_src_vertex_; i++)
		{
			int* out_indices = new int[knn_num_neighbor+1];
        	Scalar *out_dists = new Scalar[knn_num_neighbor+1];
			src_tree->query(src_points_.col(i).data(), knn_num_neighbor+1, out_indices, out_dists);
			for(int j = 0; j < knn_num_neighbor; j++)
			{
				src_knn_indices_(j, i) = out_indices[j+1];
			}
			delete[] out_indices;
			delete[] out_dists;
		}
		delete src_tree;
	}

    Timer::EventID time1, time2;
    time1 = timer.get_time();
	Scalar sample_radius;
	if(pars_.use_coarse_reg)
	{
		if(pars_.use_geodesic_dist)
			sample_radius = src_sample_nodes.SampleAndConstuct(*src_mesh_, pars_.uni_sample_radio,  src_points_); 
		else
			sample_radius = src_sample_nodes.SampleAndConstuctFPS(*src_mesh_, pars_.uni_sample_radio, src_points_, src_knn_indices_, 4, 8);
	}

    time2 = timer.get_time();
	#ifdef DEBUG
	std::cout << "construct deformation graph time = " << timer.elapsed_time(time1, time2) << std::endl;
	#endif

	#ifdef DEBUG
	if(pars_.use_coarse_reg)
	{
        std::string out_node = "test.obj"; // pars_.out_each_step_info;
        src_sample_nodes.print_nodes(*src_mesh_, out_node);//init sample nodes
	}
    #endif

	if(pars_.use_coarse_reg)
	{
		num_sample_nodes = src_sample_nodes.nodeSize();
		pars_.num_sample_nodes = num_sample_nodes;

		X_.resize(12 * num_sample_nodes); X_.setZero();
		align_coeff_PV0_.resize(3 * n_src_vertex_, 12 * num_sample_nodes);
		nodes_P_.resize(n_src_vertex_ * 3);

		nodes_R_.resize(9 * num_sample_nodes); nodes_R_.setZero();
		rigid_coeff_L_.resize(12 * num_sample_nodes, 12 * num_sample_nodes);
		rigid_coeff_J_.resize(12 * num_sample_nodes, 9 * num_sample_nodes);

		std::vector<Triplet> coeffv(4 * num_sample_nodes);
		std::vector<Triplet> coeffL(9 * num_sample_nodes);
		std::vector<Triplet> coeffJ(9 * num_sample_nodes);
		for (int i = 0; i < num_sample_nodes; i++)
		{
			// X_
			X_[12 * i] = 1.0;
			X_[12 * i + 4] = 1.0;
			X_[12 * i + 8] = 1.0;

			// nodes_R_
			nodes_R_[9 * i] = 1.0;
			nodes_R_[9 * i + 4] = 1.0;
			nodes_R_[9 * i + 8] = 1.0;

			for (int j = 0; j < 9; j++)
			{
				// rigid_coeff_L_
				coeffL[9 * i + j] = Triplet(12 * i + j, 12 * i + j, 1.0);
				// rigid_coeff_J_
				coeffJ[9 * i + j] = Triplet(12 * i + j, 9 * i + j, 1.0);
			}
		}
		rigid_coeff_L_.setFromTriplets(coeffL.begin(), coeffL.end());
		rigid_coeff_J_.setFromTriplets(coeffJ.begin(), coeffJ.end());
		
		// 0.02s
		// update coefficient matrices
		src_sample_nodes.initWeight(align_coeff_PV0_, nodes_P_, 
			reg_coeff_B_, reg_right_D_, reg_cwise_weights_);
		
		num_graph_edges = reg_cwise_weights_.rows();
	
	}
	
	// update ARAP coeffs
	FullInARAPCoeff();

	local_rotations_.resize(3, n_src_vertex_ * 3);

	if(pars_.use_geodesic_dist)
	{
		num_edges = src_mesh_->n_halfedges();
		arap_right_.resize(3*src_mesh_->n_halfedges());
		arap_right_fine_.resize(3 * src_mesh_->n_halfedges());
	}
	else{
		num_edges = knn_num_neighbor*n_src_vertex_;
		arap_right_.resize(3*knn_num_neighbor*n_src_vertex_);
		arap_right_fine_.resize(3 * knn_num_neighbor*n_src_vertex_);
	}
	

	
	InitRotations();

	target_normals_.resize(3, n_tar_vertex_);
	for (int i = 0; i < n_tar_vertex_; i++)
	{
		Vec3 n = tar_mesh_->normal(tar_mesh_->vertex_handle(i));
		target_normals_(0, i) = n[0];
		target_normals_(1, i) = n[1];
		target_normals_(2, i) = n[2];
	}	

	
	Timer::EventID begin_sampling, end_sampling;
    begin_sampling = timer.get_time();

	sampling_indices_.clear();

    // start points
    size_t startIndex = 0;
    sampling_indices_.push_back(startIndex);


	// FPS to get sampling points in align term
	VectorX minDistances(n_src_vertex_);
	minDistances.setConstant(std::numeric_limits<Scalar>::max());
	minDistances[startIndex] = 0;

	vertex_sample_indices_.resize(n_src_vertex_, -1);
	vertex_sample_indices_[startIndex] = 0;
	
    // repeat select farthest points
    while (sampling_indices_.size() < align_sampling_num_) {
        // calculate the distance between each point with the sampling points set.         
		#pragma omp parallel for
        for (size_t i = 0; i < n_src_vertex_; ++i) {
			if(i==startIndex)
				continue;
			
			Scalar dist = (src_points_.col(startIndex) - src_points_.col(i)).norm();
			if(dist < minDistances[i])
				minDistances[i] = dist;
        }

        // choose farthest point
		int maxDistanceIndex;
		minDistances.maxCoeff(&maxDistanceIndex);
		minDistances[maxDistanceIndex] = 0;

        // add the farthest point into the sampling points set.
        sampling_indices_.push_back(maxDistanceIndex);
		startIndex= maxDistanceIndex;
		vertex_sample_indices_[startIndex] = sampling_indices_.size()-1;
    }
	
	end_sampling = timer.get_time();
	#ifdef DEBUG
	std::cout << "cur_sample_idx = " << sampling_indices_.size() << " time = " << timer.elapsed_time(begin_sampling, end_sampling) << std::endl;
	#endif
}

void NonRigidreg::InitWelschParam()
{
    // welsch parameters
    weight_d_.resize(n_src_vertex_*3);
	weight_d_.setOnes();

    // Initialize correspondences
    InitCorrespondence(correspondence_pairs_);

    VectorX init_nus(correspondence_pairs_.size());
	#pragma omp parallel for
    for(size_t i = 0; i < correspondence_pairs_.size(); i++)
    {
        Vector3 closet = correspondence_pairs_[i].position;
        init_nus[i] = (src_mesh_->point(src_mesh_->vertex_handle(correspondence_pairs_[i].src_idx))
                    - Vec3(closet[0], closet[1], closet[2])).norm();
    }
    igl::median(init_nus, pars_.Data_nu);

    if(pars_.calc_gt_err&&n_src_vertex_ == n_tar_vertex_)
    {
        VectorX gt_err(n_src_vertex_);
        for(int i = 0; i < n_src_vertex_; i++)
        {
            gt_err[i] = (src_mesh_->point(src_mesh_->vertex_handle(i)) - tar_mesh_->point(tar_mesh_->vertex_handle(i))).norm();
        }
        pars_.init_gt_mean_errs = std::sqrt(gt_err.squaredNorm()/n_src_vertex_);
        pars_.init_gt_max_errs = gt_err.maxCoeff();
    }
}


Scalar NonRigidreg::DoNonRigid()
{
    // Data term parameters
    Scalar nu1 = pars_.Data_initk * pars_.Data_nu;

	if(pars_.use_coarse_reg)
	{
		optimize_w_align = 1.0;
		optimize_w_smo = pars_.w_smo/reg_coeff_B_.rows() * sampling_indices_.size(); 
		optimize_w_rot = pars_.w_rot/num_sample_nodes * sampling_indices_.size();
		optimize_w_arap = pars_.w_arap_coarse/arap_coeff_.rows() * sampling_indices_.size();
		std::cout << "Coarse Stage: optimize w_align = " << optimize_w_align << " | w_smo = " << optimize_w_smo << " | w_rot = " << optimize_w_rot << " | w_arap = " << optimize_w_arap << std::endl;
		GraphCoarseReg(nu1);
	}
	else
	{
		std::cout << "Coarse registration does not apply!" << std::endl;
	}

	
	if(pars_.use_fine_reg)
	{
		optimize_w_align = 1.0;
		optimize_w_arap = pars_.w_arap_fine /arap_coeff_fine_.rows() * n_src_vertex_;
		std::cout << "Fine Stage: optimize w_align = " << optimize_w_align << " | w_arap = " << optimize_w_arap << std::endl;
		PointwiseFineReg(nu1);
	}
	else{
		std::cout << "Fine registration does not apply!" << std::endl;
	}

	Scalar gt_err = SetMeshPoints(src_mesh_, deformed_points_);
    return 0;
}

void NonRigidreg::CalcNodeRotations()
{
#pragma omp parallel for
    for (int i = 0; i < num_sample_nodes; i++)
    {
		Matrix33 rot;
        Eigen::JacobiSVD<Matrix33> svd(Eigen::Map<Matrix33>(X_.data()+12*i, 3,3), Eigen::ComputeFullU | Eigen::ComputeFullV);
        if (svd.matrixU().determinant()*svd.matrixV().determinant() < 0.0) {
            Vector3 S = Vector3::Ones(); S(2) = -1.0;
            rot = svd.matrixU()*S.asDiagonal()*svd.matrixV().transpose();
        }
        else {
            rot = svd.matrixU()*svd.matrixV().transpose();
        }
		nodes_R_.segment(9 * i, 9) = Eigen::Map<VectorX>(rot.data(), 9);
    }
}

void NonRigidreg::welsch_weight(VectorX& r, Scalar p) {
#pragma omp parallel for
    for (int i = 0; i<r.rows(); ++i) {
		if(r[i] >= 0)
        	r[i] = std::exp(-r[i] / (2 * p*p));
		else
			r[i] = 0.;
    }
}

Scalar NonRigidreg::SetMeshPoints(Mesh* mesh, const VectorX & target)
{
    VectorX gt_errs(n_src_vertex_);
#pragma omp parallel for
    for (int i = 0; i < n_src_vertex_; i++)
    {
		Vec3 p(target[i * 3], target[i * 3 + 1], target[i * 3 + 2]);
        mesh->set_point(mesh->vertex_handle(i), p);
		Vec3 n(deformed_normals_(0,i), deformed_normals_(1,i),deformed_normals_(2,i));
		mesh->set_normal(mesh->vertex_handle(i), n);
		if (pars_.calc_gt_err)
			gt_errs[i] = (target.segment(3 * i, 3) - tar_points_.col(i)).squaredNorm();
    }
    if(pars_.calc_gt_err)
        return gt_errs.sum()/n_src_vertex_;
    else
        return -1.0;
}


void NonRigidreg::FullInARAPCoeff()
{
	arap_laplace_weights_.resize(n_src_vertex_);
	Timer timer;
	
	if(pars_.use_geodesic_dist)
	{
		for (int i = 0; i < n_src_vertex_; i++)
		{
			int nn = 0;
			OpenMesh::VertexHandle vh = src_mesh_->vertex_handle(i);
			for (auto vv = src_mesh_->vv_begin(vh); vv != src_mesh_->vv_end(vh); vv++)
			{
				nn++;
			}
			arap_laplace_weights_[i] = 1.0 / nn;
		}

		std::vector<Triplet> coeffs;
		std::vector<Triplet> coeffs_fine;
		for (int i = 0; i < src_mesh_->n_halfedges(); i++)
		{
			int src_idx = src_mesh_->from_vertex_handle(src_mesh_->halfedge_handle(i)).idx();
			int tar_idx = src_mesh_->to_vertex_handle(src_mesh_->halfedge_handle(i)).idx();
			Scalar w = sqrtf(arap_laplace_weights_[src_idx]);

			if(pars_.use_coarse_reg)
			{
			for (int k = 0; k < 3; k++)
			{
				for (RowMajorSparseMatrix::InnerIterator it(align_coeff_PV0_, src_idx*3+k); it; ++it)
				{
					coeffs.push_back(Triplet(i*3+k, it.col(), w*it.value()));
				}
				for (RowMajorSparseMatrix::InnerIterator it(align_coeff_PV0_, tar_idx*3+k); it; ++it)
				{
					coeffs.push_back(Triplet(i*3+k, it.col(), -w*it.value()));
				}
			}
			}

			coeffs_fine.push_back(Triplet(i * 3, src_idx * 3, w));
			coeffs_fine.push_back(Triplet(i * 3 + 1, src_idx * 3 + 1, w));
			coeffs_fine.push_back(Triplet(i * 3 + 2, src_idx * 3 + 2, w));
			coeffs_fine.push_back(Triplet(i * 3, tar_idx * 3, -w));
			coeffs_fine.push_back(Triplet(i * 3 + 1, tar_idx * 3 + 1, -w));
			coeffs_fine.push_back(Triplet(i * 3 + 2, tar_idx * 3 + 2, -w));
		}
		
		if(pars_.use_coarse_reg)
		{
			arap_coeff_.resize(src_mesh_->n_halfedges()*3, num_sample_nodes * 12);
			arap_coeff_.setFromTriplets(coeffs.begin(), coeffs.end());
			arap_coeff_mul_ = arap_coeff_.transpose() * arap_coeff_;
		}

		arap_coeff_fine_.resize(src_mesh_->n_halfedges() * 3, n_src_vertex_ * 3);
		arap_coeff_fine_.setFromTriplets(coeffs_fine.begin(), coeffs_fine.end());
		arap_coeff_mul_fine_ = arap_coeff_fine_.transpose() * arap_coeff_fine_;
	}
	else
	{
		int nn = src_knn_indices_.rows();
		for (int i = 0; i < n_src_vertex_; i++)
		{
			arap_laplace_weights_[i] = 1.0 / nn;
		}

		std::vector<Triplet> coeffs;
		std::vector<Triplet> coeffs_fine;
		for(int src_idx = 0; src_idx < n_src_vertex_; src_idx++)
		{
			for(int j = 0; j < nn; j++)
			{
				int i = src_idx*nn+j;
				int tar_idx = src_knn_indices_(j, src_idx);
				Scalar w = sqrtf(arap_laplace_weights_[src_idx]);

				if(pars_.use_coarse_reg)
				{
				for (int k = 0; k < 3; k++)
				{
					for (RowMajorSparseMatrix::InnerIterator it(align_coeff_PV0_, src_idx*3+k); it; ++it)
					{
						coeffs.push_back(Triplet(i*3+k, it.col(), w*it.value()));
					}
					for (RowMajorSparseMatrix::InnerIterator it(align_coeff_PV0_, tar_idx*3+k); it; ++it)
					{
						coeffs.push_back(Triplet(i*3+k, it.col(), -w*it.value()));
					}
				}
				}

				coeffs_fine.push_back(Triplet(i * 3, src_idx * 3, w));
				coeffs_fine.push_back(Triplet(i * 3 + 1, src_idx * 3 + 1, w));
				coeffs_fine.push_back(Triplet(i * 3 + 2, src_idx * 3 + 2, w));
				coeffs_fine.push_back(Triplet(i * 3, tar_idx * 3, -w));
				coeffs_fine.push_back(Triplet(i * 3 + 1, tar_idx * 3 + 1, -w));
				coeffs_fine.push_back(Triplet(i * 3 + 2, tar_idx * 3 + 2, -w));
			}
		}
		if(pars_.use_coarse_reg)
		{
			arap_coeff_.resize(n_src_vertex_*nn*3, num_sample_nodes * 12);
			arap_coeff_.setFromTriplets(coeffs.begin(), coeffs.end());
			arap_coeff_mul_ = arap_coeff_.transpose() * arap_coeff_;
		}

		arap_coeff_fine_.resize(n_src_vertex_*nn * 3, n_src_vertex_ * 3);
		arap_coeff_fine_.setFromTriplets(coeffs_fine.begin(), coeffs_fine.end());
		arap_coeff_mul_fine_ = arap_coeff_fine_.transpose() * arap_coeff_fine_;
	}
}


void NonRigidreg::CalcARAPRight()
{
	if(pars_.use_geodesic_dist)
	{
		#pragma omp parallel for
		for (int i = 0; i < src_mesh_->n_halfedges(); i++)
		{
			int src_idx = src_mesh_->from_vertex_handle(src_mesh_->halfedge_handle(i)).idx();
			int tar_idx = src_mesh_->to_vertex_handle(src_mesh_->halfedge_handle(i)).idx();

			Vector3 vij = local_rotations_.block(0, 3 * src_idx,3, 3) * (src_points_.col(src_idx) - src_points_.col(tar_idx));

			Scalar w = sqrtf(arap_laplace_weights_[src_idx]);

			
			arap_right_[i * 3] = w*(vij[0] - nodes_P_[src_idx * 3] + nodes_P_[tar_idx * 3]);
			arap_right_[i * 3 + 1] = w*(vij[1] - nodes_P_[src_idx * 3 + 1] + nodes_P_[tar_idx * 3 + 1]);
			arap_right_[i * 3 + 2] = w*(vij[2] - nodes_P_[src_idx * 3 + 2] + nodes_P_[tar_idx * 3 + 2]);
		}
	}
	else
	{
		int nn = src_knn_indices_.rows();
		#pragma omp parallel for
		for (int src_idx = 0; src_idx < n_src_vertex_; src_idx++)
		{
			for (int j = 0; j < nn; j++)
			{
			int i = src_idx*nn + j;
			int tar_idx = src_knn_indices_(j, src_idx);

			Vector3 vij = local_rotations_.block(0, 3 * src_idx,3, 3) * (src_points_.col(src_idx) - src_points_.col(tar_idx));

			Scalar w = sqrtf(arap_laplace_weights_[src_idx]);

			
			arap_right_[i * 3] = w*(vij[0] - nodes_P_[src_idx * 3] + nodes_P_[tar_idx * 3]);
			arap_right_[i * 3 + 1] = w*(vij[1] - nodes_P_[src_idx * 3 + 1] + nodes_P_[tar_idx * 3 + 1]);
			arap_right_[i * 3 + 2] = w*(vij[2] - nodes_P_[src_idx * 3 + 2] + nodes_P_[tar_idx * 3 + 2]);
			}
		}
	}
}

void NonRigidreg::CalcARAPRightFine()
{
	if(pars_.use_geodesic_dist)
	{
		#pragma omp parallel for
		for (int i = 0; i < src_mesh_->n_halfedges(); i++)
		{
			int src_idx = src_mesh_->from_vertex_handle(src_mesh_->halfedge_handle(i)).idx();
			int tar_idx = src_mesh_->to_vertex_handle(src_mesh_->halfedge_handle(i)).idx();

			Vector3 vij = local_rotations_.block(0, 3 * src_idx, 3, 3) * (src_points_.col(src_idx) - src_points_.col(tar_idx));

			Scalar w = sqrtf(arap_laplace_weights_[src_idx]);

			arap_right_fine_[i * 3] = w*(vij[0]);
			arap_right_fine_[i * 3 + 1] = w*(vij[1]);
			arap_right_fine_[i * 3 + 2] = w*(vij[2]);
		}
	}
	else
	{
		int nn = src_knn_indices_.rows();
		#pragma omp parallel for
		for(int src_idx = 0; src_idx < n_src_vertex_; src_idx++)
		{
			for(int j = 0; j < nn; j++)
			{
				int tar_idx = src_knn_indices_(j, src_idx);
				int i = src_idx*nn+j;

				Vector3 vij = local_rotations_.block(0, 3 * src_idx, 3, 3) * (src_points_.col(src_idx) - src_points_.col(tar_idx));

				Scalar w = sqrtf(arap_laplace_weights_[src_idx]);

				arap_right_fine_[i * 3] = w*(vij[0]);
				arap_right_fine_[i * 3 + 1] = w*(vij[1]);
				arap_right_fine_[i * 3 + 2] = w*(vij[2]);
			}
		}
	}
}

void NonRigidreg::InitRotations()
{
	local_rotations_.resize(3, n_src_vertex_ * 3);
	local_rotations_.setZero();
#pragma omp parallel for
	for (int i = 0; i < n_src_vertex_; i++)
	{
		local_rotations_(0, i * 3) = 1;
		local_rotations_(1, i * 3 + 1) = 1;
		local_rotations_(2, i * 3 + 2) = 1;
	}
}


void NonRigidreg::CalcLocalRotations(bool isCoarseAlign)
{
	
	#pragma omp parallel for
	for (int i = 0; i < n_src_vertex_; i++)
	{
		Matrix33 sum;
		sum.setZero();

		int nn = 0;
		if(pars_.use_geodesic_dist)
		{
			OpenMesh::VertexHandle vh = src_mesh_->vertex_handle(i);
			for (auto vv = src_mesh_->vv_begin(vh); vv != src_mesh_->vv_end(vh); vv++)
			{
				int neighbor_idx = vv->idx();
				Vector3 dv = src_points_.col(i) - src_points_.col(neighbor_idx);
				Vector3 new_dv = deformed_points_.segment(3 * i, 3) - deformed_points_.segment(3 * neighbor_idx, 3);
				sum += dv * new_dv.transpose();
				nn++;
			}
		}
		else
		{
			nn = src_knn_indices_.rows();
			for(int j = 0; j < nn; j++)
			{
				int neighbor_idx = src_knn_indices_(j, i);
				Vector3 dv = src_points_.col(i) - src_points_.col(neighbor_idx);
				Vector3 new_dv = deformed_points_.segment(3 * i, 3) - deformed_points_.segment(3 * neighbor_idx, 3);
				sum += dv * new_dv.transpose();
			}
		}
		

		sum*= 1.0*optimize_w_arap/nn;
		
		if(!isCoarseAlign)
		{
			int tar_idx = correspondence_pairs_[i].tar_idx;
			Vector3 d = deformed_points_.segment(3 * i, 3) - tar_points_.col(tar_idx);
			Scalar c = (target_normals_.col(tar_idx) + deformed_normals_.col(i)).dot(d);
			Scalar d_norm2 = d[0]*d[0] + d[1]*d[1] + d[2]*d[2];
			Vector3 h = deformed_normals_.col(i) - c*d/d_norm2;

			Scalar w = optimize_w_align*d_norm2*weight_d_[i];
			sum += w * src_normals_.col(i) * h.transpose();
		}
		else if(vertex_sample_indices_[i] >= 0)
		{
			//Vector3 Rv = rotations_.block(3 * i, 0, 3, 3) * src_normals.col(i);
			// R^k n_s - d * (n_t + R^k n_s )*d / ||d||^2
			// deformed_normals = R^k n_s
			// d = v_i - u_j
			int tar_idx = correspondence_pairs_[vertex_sample_indices_[i]].tar_idx;
			Vector3 d = deformed_points_.segment(3 * i, 3) - tar_points_.col(tar_idx);
			Scalar c = (target_normals_.col(tar_idx) + deformed_normals_.col(i)).dot(d);
			Scalar d_norm2 = d[0]*d[0] + d[1]*d[1] + d[2]*d[2];
			Vector3 h = deformed_normals_.col(i) - c*d/d_norm2;

			Scalar w = optimize_w_align*d_norm2*weight_d_[vertex_sample_indices_[i]];
			sum += w * src_normals_.col(i) * h.transpose();
		}
		

		Eigen::JacobiSVD<Matrix33> svd(sum, Eigen::ComputeFullU | Eigen::ComputeFullV);

		if (svd.matrixU().determinant()*svd.matrixV().determinant() < 0.0) {
			Vector3 S = Vector3::Ones(); S(2) = -1.0;
			sum = svd.matrixV()*S.asDiagonal()*svd.matrixU().transpose();
		}
		else {
			sum = svd.matrixV()*svd.matrixU().transpose();
		}
		for (int s = 0; s < 3; s++)
		{
			for (int t = 0; t < 3; t++)
			{
				local_rotations_(s, 3 * i + t) = sum(s, t);
			}
		}
	}
}

void NonRigidreg::CalcDeformedNormals()
{
#pragma omp parallel for
	for (int i = 0; i < n_src_vertex_; i++)
	{
		deformed_normals_.col(i) = local_rotations_.block(0, i * 3, 3, 3) * src_normals_.col(i);
	}
}

void NonRigidreg::CalcNormalsSum()
{
	#pragma omp parallel for
	for(int i = 0; i < correspondence_pairs_.size(); i++)
    {
		int sidx = correspondence_pairs_[i].src_idx;
		int tidx = correspondence_pairs_[i].tar_idx;
		int j = 0;
        for (RowMajorSparseMatrix::InnerIterator it(normals_sum_, i); it; ++it)
        {
			it.valueRef() = deformed_normals_(j, sidx) + target_normals_(j, tidx);
			j++;
        }
    }
}

void NonRigidreg::InitNormalsSum()
{
	std::vector<Triplet> coeffs(3 * correspondence_pairs_.size());
	normals_sum_.resize(correspondence_pairs_.size(), 3*n_src_vertex_);
	normals_sum_.setZero();

#pragma omp parallel for
	for(int i = 0; i < correspondence_pairs_.size(); i++)
	{
		int sidx = correspondence_pairs_[i].src_idx;
		int tidx = correspondence_pairs_[i].tar_idx;
		coeffs[i * 3] = Triplet(i, 3 * sidx, deformed_normals_(0, sidx) + target_normals_(0, tidx));
		coeffs[i * 3 + 1] = Triplet(i, 3 * sidx + 1, deformed_normals_(1, sidx) + target_normals_(1, tidx));
		coeffs[i * 3 + 2] = Triplet(i, 3 * sidx + 2, deformed_normals_(2, sidx) + target_normals_(2, tidx));
	}
	normals_sum_.setFromTriplets(coeffs.begin(), coeffs.end());
}

void NonRigidreg::PointwiseFineReg(Scalar nu1)
{
	Scalar energy=-1., align_err=-1., arap_err=-1.;

	VectorX prevV = VectorX::Zero(n_src_vertex_ * 3);

	bool run_once = true;

	Timer time;
	Timer::EventID begin_time, run_time;

	// Smooth term parameters
	w_align = optimize_w_align; 
	w_smo = optimize_w_smo; 

	if (pars_.data_use_robust_weight)
	{
		w_align = optimize_w_align *(2.0*nu1*nu1);
	}

	pars_.each_energys.push_back(0.0);
	pars_.each_gt_max_errs.push_back(pars_.init_gt_max_errs);
	pars_.each_gt_mean_errs.push_back(pars_.init_gt_mean_errs);
	pars_.each_iters.push_back(0);
	pars_.each_times.push_back(pars_.non_rigid_init_time);
	pars_.each_term_energy.push_back(Vector4(0, 0, 0, 0));

	Scalar gt_err = -1;

	double construct_mat_time = 0.0;
	double solve_eq_time = 0.0;

	begin_time = time.get_time();

	int out_iter = 0;
	while (out_iter < pars_.max_outer_iters)
	{
		// Find clost points
		FindClosestPoints(correspondence_pairs_, deformed_points_);
		// according correspondence_pairs to update corres_U0_;
		corres_U0_.setZero();
		weight_d_.resize(n_src_vertex_);

		for (size_t i = 0; i < correspondence_pairs_.size(); i++)
		{
			corres_U0_.segment(i * 3, 3) = correspondence_pairs_[i].position;
			weight_d_[i] = correspondence_pairs_[i].min_dist2;
			int tar_idx = correspondence_pairs_[i].tar_idx;
			if(deformed_normals_.col(i).dot(target_normals_.col(tar_idx))<0)
				weight_d_[i] = -1;
		}

		// update weight
		if (pars_.data_use_robust_weight)
		{
			welsch_weight(weight_d_, nu1);
		}
		else
			weight_d_.setOnes();

		// int welsch_iter;
		int total_inner_iters = 0;

		if (run_once == true && pars_.use_landmark == true)
		{
			weight_d_.setOnes();
		}


		// update V,U and D
		#ifdef DEBUG
		Timer::EventID construct_mat_begin = time.get_time();
		#endif
		// construct matrix A0 and pre-decompose
		if (pars_.use_symm_ppl)
		{
			if(out_iter==0)
				InitNormalsSum();
			else
				CalcNormalsSum();
			
			RowMajorSparseMatrix normals_sum_mul = normals_sum_.transpose() * weight_d_.asDiagonal()* normals_sum_;
			mat_A0_ = optimize_w_align * normals_sum_mul
				+ optimize_w_arap * arap_coeff_mul_fine_;
			
			CalcARAPRightFine();

			vec_b_ = optimize_w_align * normals_sum_mul * corres_U0_
				+ optimize_w_arap * arap_coeff_fine_.transpose() * arap_right_fine_;
			
		}
		else
		{
			RowMajorSparseMatrix diag_weights;
			diag_weights.resize(n_src_vertex_ * 3, n_src_vertex_ * 3);
			std::vector<Triplet> coeffs_diag_weights(n_src_vertex_ * 3);

			VectorX weight_corres_U(n_src_vertex_*3);
			
			for(int i = 0; i < n_src_vertex_; i++)
			{
				coeffs_diag_weights[i*3] = Triplet(i*3, i*3, weight_d_[i]);
				coeffs_diag_weights[i*3+1] = Triplet(i*3+1, i*3+1, weight_d_[i]);
				coeffs_diag_weights[i*3+2] = Triplet(i*3+2, i*3+2, weight_d_[i]);

				weight_corres_U[i*3] = weight_d_[i] * corres_U0_[i*3];
				weight_corres_U[i*3+1] = weight_d_[i] * corres_U0_[i*3+1];
				weight_corres_U[i*3+2] = weight_d_[i] * corres_U0_[i*3+2];
			}
			diag_weights.setFromTriplets(coeffs_diag_weights.begin(), coeffs_diag_weights.end());

			mat_A0_ = optimize_w_align * diag_weights
				+ optimize_w_arap * arap_coeff_mul_fine_;
			
			CalcARAPRightFine();
			vec_b_ = optimize_w_align * weight_corres_U
				+ optimize_w_arap * arap_coeff_fine_.transpose() * arap_right_fine_;
		}

		
		#ifdef DEBUG
		Timer::EventID construct_mat_end = time.get_time();
		construct_mat_time += time.elapsed_time(construct_mat_begin, construct_mat_end);
		#endif
		
		if (run_once)
		{
			solver_.analyzePattern(mat_A0_);
			run_once = false;
		}
		solver_.factorize(mat_A0_);
		deformed_points_ = solver_.solve(vec_b_);

		run_time = time.get_time();
		double eps_time = time.elapsed_time(begin_time, run_time);
		pars_.each_times.push_back(eps_time);

		#ifdef DEBUG
		solve_eq_time += time.elapsed_time(construct_mat_end, run_time);
		energy = CalcEnergyFine(align_err, arap_err);
		#endif

		
		CalcLocalRotations(false);
		CalcDeformedNormals();
		
		#ifdef DEBUG
		if (n_src_vertex_ == n_tar_vertex_)
			gt_err = (deformed_points_ - Eigen::Map<VectorX>(tar_points_.data(), 3 * n_src_vertex_)).squaredNorm();
		#endif

		// save results
		pars_.each_gt_mean_errs.push_back(gt_err);
		pars_.each_gt_max_errs.push_back(0);
		pars_.each_energys.push_back(energy);
		pars_.each_iters.push_back(total_inner_iters);
		pars_.each_term_energy.push_back(Vector4(align_err, 0, 0, arap_err));

		if((deformed_points_ - prevV).norm()/sqrtf(n_src_vertex_) < pars_.stop_fine)
		{
			break;
		}
		prevV = deformed_points_;
		out_iter++;
	}

	#ifdef DEBUG
	std::cout << "construct mat time = " << construct_mat_time 
	<< "\nsolve_eq time = " << solve_eq_time  << " iters = " << out_iter << std::endl;
	#endif
}

void NonRigidreg::GraphCoarseReg(Scalar nu1)
{
	Scalar energy=0., align_err=0., reg_err=0., rot_err=0., arap_err=0.;
	VectorX prevV = VectorX::Zero(n_src_vertex_ * 3);

	// welsch_sweight
 	bool run_once = true;

	Timer time;
	Timer::EventID begin_time, run_time;
	pars_.each_energys.clear();
	pars_.each_gt_mean_errs.clear();
	pars_.each_gt_max_errs.clear();
	pars_.each_times.clear();
	pars_.each_iters.clear();
	pars_.each_term_energy.clear();

	w_align = optimize_w_align; 
	w_smo = optimize_w_smo; 

	if (pars_.data_use_robust_weight)
	{
		w_align = optimize_w_align *(2.0*nu1*nu1);
	}

	pars_.each_energys.push_back(0.0);
	pars_.each_gt_max_errs.push_back(pars_.init_gt_max_errs);
	pars_.each_gt_mean_errs.push_back(pars_.init_gt_mean_errs);
	pars_.each_iters.push_back(0);
	pars_.each_times.push_back(pars_.non_rigid_init_time);
	pars_.each_term_energy.push_back(Vector4(0, 0, 0, 0));

	Scalar gt_err;

	VectorX prev_X = X_;

	begin_time = time.get_time();
	
	#ifdef DEBUG
	double find_cp_time = 0.0;
	double construct_mat_time = 0.0;
	double solve_eq_time = 0.0;
	double calc_energy_time = 0.0;
	double robust_weight_time = 0.0;
	double update_r_time = 0.0;
	#endif


		
	RowMajorSparseMatrix A_fixed_coeff = optimize_w_smo * reg_coeff_B_.transpose() * reg_cwise_weights_.asDiagonal() * reg_coeff_B_  + optimize_w_rot * rigid_coeff_L_ + optimize_w_arap * arap_coeff_mul_;
	int out_iter = 0;
	while (out_iter < pars_.max_outer_iters)
	{
		#ifdef DEBUG
		Timer::EventID inner_start_time = time.get_time();
		#endif
		
		correspondence_pairs_.clear();
		FindClosestPoints(correspondence_pairs_, deformed_points_, sampling_indices_);
		corres_U0_.setZero();
		weight_d_.resize(correspondence_pairs_.size());
		weight_d_.setConstant(-1);

#pragma omp parallel for
		for (size_t i = 0; i < correspondence_pairs_.size(); i++)
		{
			corres_U0_.segment(correspondence_pairs_[i].src_idx * 3, 3) = correspondence_pairs_[i].position;
			weight_d_[i] = correspondence_pairs_[i].min_dist2;
			if(deformed_normals_.col(correspondence_pairs_[i].src_idx).dot(target_normals_.col(correspondence_pairs_[i].tar_idx))<0)
				weight_d_[i] = -1;
		}


		#ifdef DEBUG
		Timer::EventID end_find_cp = time.get_time();
		double eps_time1 = time.elapsed_time(inner_start_time, end_find_cp);
		find_cp_time += eps_time1;
		#endif

		// update weight
		if (pars_.data_use_robust_weight)
		{
			welsch_weight(weight_d_, nu1);
		}
		else
		{
			weight_d_.setOnes();
		}
		
		// int welsch_iter;
		int total_inner_iters = 0;

		if (run_once == true && pars_.use_landmark == true)
		{
			weight_d_.setOnes();
		}


		#ifdef DEBUG
			Timer::EventID end_robust_weight = time.get_time();
			eps_time1 = time.elapsed_time(end_find_cp, end_robust_weight);
			robust_weight_time += eps_time1;
			#endif

		if(pars_.use_symm_ppl)
		{			
			// 0.5s (0.01s / 50 iters)
			if(out_iter==0)					
				InitNormalsSum();
			else
				CalcNormalsSum();
			
			// 1e-3
			diff_UP_ = (corres_U0_ - nodes_P_);

			// 0.37s 
			RowMajorSparseMatrix weight_NPV = normals_sum_ * align_coeff_PV0_;

			mat_A0_ = optimize_w_align * weight_NPV.transpose() * weight_d_.asDiagonal() *  weight_NPV + A_fixed_coeff; 

			CalcARAPRight();
			vec_b_ = optimize_w_align * weight_NPV.transpose() * weight_d_.asDiagonal() * normals_sum_ * diff_UP_ + optimize_w_smo * reg_coeff_B_.transpose() * reg_cwise_weights_.asDiagonal() * reg_right_D_ + optimize_w_rot * rigid_coeff_J_ * nodes_R_ + optimize_w_arap * arap_coeff_.transpose() * arap_right_;

		}
		else
		{
			VectorX weight_d3(3*n_src_vertex_);
			weight_d3.setZero();
			for(int i = 0; i < n_src_vertex_; i++)
			{
				int idx = vertex_sample_indices_[i];
				if(idx>=0)
					weight_d3[i*3] = weight_d3[i*3+1] = weight_d3[i*3+2] = weight_d_[idx];
			}

			diff_UP_ = (corres_U0_ - nodes_P_);

			mat_A0_ = optimize_w_align *align_coeff_PV0_.transpose() * weight_d3.asDiagonal() * align_coeff_PV0_  + A_fixed_coeff; 
			
			CalcARAPRight();
			vec_b_ = optimize_w_align * align_coeff_PV0_.transpose() * weight_d3.asDiagonal() * diff_UP_ + optimize_w_smo * reg_coeff_B_.transpose() * reg_cwise_weights_.asDiagonal() * reg_right_D_ + optimize_w_rot * rigid_coeff_J_ * nodes_R_ + optimize_w_arap * arap_coeff_.transpose() * arap_right_;


		}

		#ifdef DEBUG
		Timer::EventID end_construct_eq = time.get_time();
		eps_time1 = time.elapsed_time(end_robust_weight, end_construct_eq);
		construct_mat_time += eps_time1;	
		#endif

		if (run_once)
		{
			solver_.analyzePattern(mat_A0_);
			run_once = false;
		}
		solver_.factorize(mat_A0_);
		X_ = solver_.solve(vec_b_);		

		run_time = time.get_time();
		double eps_time = time.elapsed_time(begin_time, run_time);
		pars_.each_times.push_back(eps_time);

		#ifdef DEBUG
		eps_time1 = time.elapsed_time(end_construct_eq, run_time);
		solve_eq_time += eps_time1;
		#endif


		#ifdef DEBUG
		energy = CalcEnergy(align_err, reg_err, rot_err, arap_err, reg_cwise_weights_);
		#endif 

		deformed_points_ = align_coeff_PV0_ * X_ + nodes_P_;

		
		#ifdef DEBUG
		Timer::EventID end_calc_energy = time.get_time();
		eps_time1 = time.elapsed_time(run_time, end_calc_energy);
		calc_energy_time += eps_time1;
		#endif

		CalcLocalRotations(true);

		#ifdef DEBUG
		Timer::EventID end_update_r = time.get_time();
		eps_time1 = time.elapsed_time(end_calc_energy, end_update_r);
		update_r_time += eps_time1;
		#endif

		CalcNodeRotations();
		CalcDeformedNormals();

		if (n_src_vertex_ == n_tar_vertex_)
			gt_err = (deformed_points_ - Eigen::Map<VectorX>(tar_points_.data(), 3 * n_src_vertex_)).squaredNorm();

		// save results
		pars_.each_gt_mean_errs.push_back(gt_err);
		pars_.each_gt_max_errs.push_back(0);
		pars_.each_energys.push_back(energy);
		pars_.each_iters.push_back(total_inner_iters);
		pars_.each_term_energy.push_back(Vector4(align_err, reg_err, rot_err, arap_err));

		if((deformed_points_ - prevV).norm()/sqrtf(n_src_vertex_) < pars_.stop_coarse)
		{
			break;
		}
		prevV = deformed_points_;
		out_iter++;

		#ifdef DEBUG
		Timer::EventID end_find_cp2 = time.get_time();
		eps_time1 = time.elapsed_time(end_calc_energy, end_find_cp2);
		find_cp_time += eps_time1;
		#endif
	}

	#ifdef DEBUG
	std::cout << "find cp time = " << find_cp_time
	 << "\nconstruct_mat_timem = " << construct_mat_time
	 << "\nsolve_eq_time = " << solve_eq_time
	 << "\ncalc_energy_time = " << calc_energy_time
	 << "\nrobust_weight_time = " << robust_weight_time
	 << "\nupdate_r_time = " << update_r_time
	 << "\nacculate_iter = " << out_iter << std::endl;
	#endif
}

Scalar NonRigidreg::CalcEnergy(Scalar& E_align, Scalar& E_reg,
	Scalar& E_rot, Scalar& E_arap, VectorX & reg_weights)
{
	if(pars_.use_symm_ppl)
		E_align = (normals_sum_ * (align_coeff_PV0_ * X_ - diff_UP_)).squaredNorm();
	else
		E_align = ((align_coeff_PV0_ * X_ - diff_UP_)).squaredNorm();
	
	E_reg = (reg_coeff_B_ * X_ - reg_right_D_).squaredNorm();
	E_arap = (arap_coeff_ * X_ - arap_right_).squaredNorm();
	E_rot = (rigid_coeff_J_.transpose() * X_ - nodes_R_).squaredNorm();
	
	Scalar energy = w_align * E_align 
		+ w_smo * E_reg 
		+ optimize_w_arap * E_arap 
		+ optimize_w_rot * E_rot;
	return energy;
}

Scalar NonRigidreg::CalcEnergyFine(Scalar & E_align, Scalar & E_arap)
{
	if(pars_.use_symm_ppl)
		E_align = (normals_sum_ * (deformed_points_ - corres_U0_)).squaredNorm();
	else
		E_align = ((deformed_points_ - corres_U0_)).squaredNorm();
	E_arap = (arap_coeff_fine_ * deformed_points_  - arap_right_fine_).squaredNorm();

	Scalar energy = w_align * E_align
		+ optimize_w_arap * E_arap;
	return energy;
}

// *type: 0 :median, 1: average
Scalar NonRigidreg::CalcEdgelength(Mesh* mesh, int type)
{
    Scalar med;
    if(mesh->n_faces() > 0)
    {
        VectorX edges_length(mesh->n_edges());
        for(size_t i = 0; i < mesh->n_edges();i++)
        {
            OpenMesh::VertexHandle vi = mesh->from_vertex_handle(mesh->halfedge_handle(mesh->edge_handle(i),0));
            OpenMesh::VertexHandle vj = mesh->to_vertex_handle(mesh->halfedge_handle(mesh->edge_handle(i),0));
            edges_length[i] = (mesh->point(vi) - mesh->point(vj)).norm();
        }
        if (type == 0)
            igl::median(edges_length, med);
        else
            med = edges_length.mean();
    }
    else
    {
        // source is mesh, target may be point cloud.
		int nn = src_knn_indices_.rows();
		VectorX edges_length(n_src_vertex_*nn);
        for(int src_idx = 0; src_idx < n_src_vertex_; src_idx++)
		{
			for(int j = 0; j < nn; j++)
			{
				int tar_idx = src_knn_indices_(j, src_idx);
				Scalar dist = (src_points_.col(src_idx) - src_points_.col(tar_idx)).norm();
				edges_length[src_idx*nn+j] = dist;
			}
		}
        if (type == 0)
            igl::median(edges_length, med);
        else
            med = edges_length.mean();
    }
    return med;
}

void nonRigid_spare::Reg(const std::string& file_target,
                       const std::string& file_source,
                       const std::string& out_path )
{

    Mesh src_mesh;
    Mesh tar_mesh;
	std::string tar_file;
	std::string src_file;
    std::string out_file,outpath;
	tar_file=file_target;
	src_file=file_source;
	outpath=out_path;
    std::string landmark_file;
    RegParas paras;
    Scalar input_w_smo = 0.01;
	Scalar input_w_rot = 1e-4; 
    Scalar input_radius = 10;
	Scalar input_w_arap_coarse = 500;
    Scalar input_w_arap_fine = 200;
    bool normalize = true; 

    paras.stop_coarse = 1e-3;
    paras.stop_fine = 1e-4;
    paras.max_outer_iters = 30;
	paras.use_symm_ppl = true;
    paras.use_fine_reg = true;
    paras.use_coarse_reg = true;
    paras.use_geodesic_dist = true;

    // Setting paras
    paras.w_smo = input_w_smo;
    paras.w_rot = input_w_rot;
	paras.w_arap_coarse = input_w_arap_coarse;
    paras.w_arap_fine = input_w_arap_fine;

    paras.rigid_iters = 0;
    paras.calc_gt_err = true;
    paras.uni_sample_radio = input_radius;

    paras.print_each_step_info = false;
    out_file = outpath + "_res.ply";
    std::string out_info = outpath + "_params.txt"; 
    paras.out_each_step_info = outpath; 

    read_data(src_file, src_mesh);
    read_data(tar_file, tar_mesh);
    if(src_mesh.n_vertices()==0 || tar_mesh.n_vertices()==0)
        exit(0);

    if(src_mesh.n_vertices() != tar_mesh.n_vertices())
        paras.calc_gt_err = false;

    if(src_mesh.n_faces()==0)
        paras.use_geodesic_dist = false;

    if(paras.use_landmark)
        read_landmark(landmark_file.c_str(), paras.landmark_src, paras.landmark_tar);
	
    double scale = 1;
    if(normalize)
        scale = mesh_scaling(src_mesh, tar_mesh);

    paras.mesh_scale = scale;
    NonRigidreg* reg;
    reg = new NonRigidreg;

    Timer time;
    std::cout << "registration to initial... (mesh scale: " << scale << ")" << std::endl;
    Timer::EventID time1 = time.get_time();
    reg->InitFromInput(src_mesh, tar_mesh, paras);
    // non-rigid initialize
    reg->Initialize();
    Timer::EventID time2 = time.get_time();
    reg->pars_.non_rigid_init_time = time.elapsed_time(time1, time2);
    std::cout << "non-rigid registration... (graph node number: " << reg->pars_.num_sample_nodes << ")" << std::endl;

    reg->DoNonRigid();

    Timer::EventID time3 = time.get_time();

    std::cout << "Registration done!\ninitialize time : "
              << time.elapsed_time(time1, time2) << " s \tnon-rigid reg running time = " << time.elapsed_time(time2, time3) << " s" << std::endl;
    write_data(out_file.c_str(), src_mesh, scale);

    std::cout<< "write the result to " << out_file << "\n" << std::endl;

    
    reg->pars_.print_params(out_info);

    delete reg;
}

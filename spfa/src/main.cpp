#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

inline int ravel(int i, int j, int num_cols) {
    return i * num_cols + j;
}

py::tuple spfa(py::array_t<bool> input_map, std::tuple<int, int> source) {
    const float eps = 1e-6;
    const int num_dirs = 8;
    const int dirs[num_dirs][2] = {{-1, -1}, {-1, 0}, {-1, 1}, {0, 1}, {1, 1}, {1, 0}, {1, -1}, {0, -1}};
    const float dir_lengths[num_dirs] = {std::sqrt(2.0f), 1, std::sqrt(2.0f), 1, std::sqrt(2.0f), 1, std::sqrt(2.0f), 1};

    // Process input map
    py::buffer_info map_buf = input_map.request();
    int num_rows = map_buf.shape[0];
    int num_cols = map_buf.shape[1];
    bool* map_ptr = (bool *) map_buf.ptr;

    // Get source coordinates
    int source_i = std::get<0>(source);
    int source_j = std::get<1>(source);

    int max_num_verts = num_rows * num_cols;
    int max_edges_per_vert = num_dirs;
    const float inf = 2 * max_num_verts;
    int queue_size = num_dirs * num_rows * num_cols;

    // Initialize arrays
    int* edges = new int[max_num_verts * max_edges_per_vert]();
    int* edge_counts = new int[max_num_verts]();
    int* queue = new int[queue_size]();
    bool* in_queue = new bool[max_num_verts]();
    float* weights = new float[max_num_verts * max_edges_per_vert]();
    float* dists = new float[max_num_verts];
    for (int i = 0; i < max_num_verts; ++i)
        dists[i] = inf;
    int* parents = new int[max_num_verts]();
    for (int i = 0; i < max_num_verts; ++i)
        parents[i] = -1;

    // Build graph
    for (int i = 0; i < num_rows; ++i) {
        for (int j = 0; j < num_cols; ++j) {
            int v = ravel(i, j, num_cols);
            if (!map_ptr[v])
                continue;
            for (int k = 0; k < num_dirs; ++k) {
                int ip = i + dirs[k][0], jp = j + dirs[k][1];
                if (ip < 0 || jp < 0 || ip >= num_rows || jp >= num_cols)
                    continue;
                int vp = ravel(ip, jp, num_cols);
                if (!map_ptr[vp])
                    continue;
                int e = ravel(v, edge_counts[v], max_edges_per_vert);
                edges[e] = vp;
                weights[e] = dir_lengths[k];
                edge_counts[v]++;
            }
        }
    }

    // SPFA
    int s = ravel(source_i, source_j, num_cols);
    int head = 0, tail = 0;
    dists[s] = 0;
    queue[++tail] = s;
    in_queue[s] = true;
    while (head < tail) {
        int u = queue[++head];
        in_queue[u] = false;
        for (int j = 0; j < edge_counts[u]; ++j) {
            int e = ravel(u, j, max_edges_per_vert);
            int v = edges[e];
            float new_dist = dists[u] + weights[e];
            if (new_dist < dists[v]) {
                parents[v] = u;
                dists[v] = new_dist;
                if (!in_queue[v]) {
                    assert(tail < queue_size);
                    queue[++tail] = v;
                    in_queue[v] = true;
                    if (dists[queue[tail]] < dists[queue[head + 1]])
                        std::swap(queue[tail], queue[head + 1]);
                }
            }
        }
    }

    // Copy output into numpy array
    auto output_dists = py::array_t<float>({num_rows, num_cols});
    auto output_parents = py::array_t<int>({num_rows, num_cols});
    py::buffer_info dists_buf = output_dists.request(), parents_buf = output_parents.request();
    float* dists_ptr = (float *) dists_buf.ptr;
    int* parents_ptr = (int *) parents_buf.ptr;
    for (int i = 0; i < num_rows; ++i) {
        for (int j = 0; j < num_cols; ++j) {
            int u = ravel(i, j, num_cols);
            dists_ptr[u] = (dists[u] < inf - eps) * dists[u];
            parents_ptr[u] = parents[u];
        }
    }

    // Free memory
    delete[] edges;
    delete[] edge_counts;
    delete[] queue;
    delete[] in_queue;
    delete[] weights;
    delete[] dists;
    delete[] parents;

    return py::make_tuple(output_dists, output_parents);
}

PYBIND11_MODULE(spfa, m) {
    m.doc() = R"pbdoc(
        SPFA implemented in C++
    )pbdoc";

    m.def("spfa", &spfa, R"pbdoc(
        spfa
    )pbdoc");

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}

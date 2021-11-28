#include "bfs.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cstddef>
#include <omp.h>
#include <assert.h>

#include "../common/CycleTimer.h"
#include "../common/graph.h"

#define ROOT_NODE_ID 0
#define NOT_VISITED_MARKER -1

inline void vertex_set_clear(vertex_set* list) {
    list->count = 0;
}

inline void vertex_set_init(vertex_set* list, int count) {
    list->max_vertices = count;
    list->vertices = (int*)malloc(sizeof(int) * list->max_vertices);
    vertex_set_clear(list);
}

inline void vertex_set_list_clear(vertex_set*& set_list, const int list_size) {
    #pragma omp parallel for
    for (int i = 0 ; i < list_size; i++){
        set_list[i].count = 0;
    }
}

inline void vertex_set_list_init(vertex_set*& set_list, const int list_size, const int count) {
    #pragma omp parallel for
    for (int i = 0 ; i < list_size; i++){
        vertex_set_init(&set_list[i], count);
    }
}

inline void top_down_step(
    Graph g,
    vertex_set*& frontier_list,
    int* distances,
    const int max_threads)
{
    // Run one step in top down approach

    // aliasing, frontier_list[max_threads] store the current frontier
    vertex_set& current_frontier = frontier_list[max_threads];

    // For each thread, write to its own frontier
    #pragma omp parallel for
    for (int i = 0; i < current_frontier.count; i ++) {
        const int thread_id = omp_get_thread_num();
        int node = current_frontier.vertices[i];

        int start_edge = g->outgoing_starts[node];
        int end_edge = (node == g->num_nodes - 1)
                           ? g->num_edges
                           : g->outgoing_starts[node + 1];

        // attempt to add all neighbors to the new frontier
        for (int neighbor=start_edge; neighbor<end_edge; neighbor++) {
            int outgoing = g->outgoing_edges[neighbor];

            if (distances[outgoing] == NOT_VISITED_MARKER) {
                if (__sync_bool_compare_and_swap(distances + outgoing, NOT_VISITED_MARKER, distances[node] + 1)) {
                    int current_index = frontier_list[thread_id].count++;
                    frontier_list[thread_id].vertices[current_index] = outgoing;
                }
            }
        }
    }
    // Collect the frontiers
    // TODO: parallelize with atomic add on total_count
    int total_count = 0;
    for (int i = 0; i < max_threads; i ++) {
        memcpy(current_frontier.vertices + total_count, frontier_list[i].vertices, 
                frontier_list[i].count * sizeof(int));
        total_count += frontier_list[i].count;
    }
    current_frontier.count = total_count;
}

// Implements top-down BFS.
//
// Result of execution is that, for each node in the graph, the
// distance to the root is stored in sol.distances.
void bfs_top_down(Graph graph, solution* sol) {

    const int max_threads {omp_get_max_threads()};

    const int vertex_set_list_size = max_threads+1;

    vertex_set* frontier_list = new vertex_set[vertex_set_list_size];   // first max_threads used for parallel processing, last one used for current_frontier
    vertex_set_list_init(frontier_list, vertex_set_list_size, graph->num_nodes);

    // initialize all nodes to NOT_VISITED
    #pragma omp parallel for
    for (int i = 0; i < graph->num_nodes; i ++)
        sol->distances[i] = NOT_VISITED_MARKER;

 
    // Alias
    vertex_set* frontier = &(frontier_list[max_threads]);
    // setup frontier with the root node
    frontier->vertices[frontier->count++] = ROOT_NODE_ID;
    sol->distances[ROOT_NODE_ID] = 0;

    while (frontier->count != 0) {

#ifdef VERBOSE
        double start_time = CycleTimer::currentSeconds();
#endif
        // Clear all except the last in frontier_list
        vertex_set_list_clear(frontier_list, max_threads);
        top_down_step(graph, frontier_list, sol->distances, max_threads);
#ifdef VERBOSE
    double end_time = CycleTimer::currentSeconds();
    printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
#endif
    }
    delete frontier_list;
}

inline void tracker_list_reset(bool* next_frontier_tracker, int list_len) {
    #pragma omp parallel for
    for (int i = 0; i < list_len; i ++)
        next_frontier_tracker[i] = false;
}

inline int bottom_up_step(
    Graph g,
    bool* current_frontier,
    bool* next_frontier,
    int* distances, 
    const int exploring_distance, 
    const int max_threads,
    const int schedule_chunk = 64)  // assuming 64 Byte Cache Line and bool tracker
{
    int new_node_count = 0;
    // Run one step in bottom up approach
    #pragma omp parallel for reduction(+:new_node_count) num_threads (max_threads) schedule(dynamic, schedule_chunk)
    for(int node = 0; node < g->num_nodes; node ++) {
        if (distances[node] == NOT_VISITED_MARKER) {
            //explore all incoming edges to node
            int start_edge = g->incoming_starts[node];
            int end_edge = (node == g->num_nodes - 1)
                               ? g->num_edges
                               : g->incoming_starts[node + 1];
            for (int edge = start_edge; edge < end_edge; edge ++) {
                int neighbor = g->incoming_edges[edge];
                if (current_frontier[neighbor]) {
                    distances[node] = exploring_distance;
                    next_frontier[node] = true;
                    new_node_count += 1;
                    break;
                }
            }
        }
    } 
    // printf("Step: %-10d %-10d sec\n", exploring_distance, new_node_count);
    return new_node_count; 
}

void bfs_bottom_up(Graph graph, solution* sol)
{
    // CS149 students:
    //
    // You will need to implement the "bottom up" BFS here as
    // described in the handout.
    //
    // As a result of your code's execution, sol.distances should be
    // correctly populated for all nodes in the graph.
    //
    // As was done in the top-down case, you may wish to organize your
    // code by creating subroutine bottom_up_step() that is called in
    // each step of the BFS process.    

    const int max_threads {omp_get_max_threads()};
    int chunk_size = 1024;  // assuming 64 byte cache line and 1 byte bool

    const int num_nodes = graph->num_nodes;

    while (num_nodes < max_threads * chunk_size) {
        chunk_size /= 2;
    }

    
    printf("# Threads=%-10d\n", max_threads);
    printf("Chunk size=%-10d\n", chunk_size);
    printf("# Nodes=%-10d\n", num_nodes);
    
    bool* current_frontier = new bool[graph->num_nodes];
    bool* next_frontier = new bool[graph->num_nodes];

    tracker_list_reset(current_frontier, num_nodes);

    // initialize all nodes to NOT_VISITED
    #pragma omp parallel for
    for (int i = 0; i < num_nodes; i ++)
        sol->distances[i] = NOT_VISITED_MARKER;

    int frontier_count = 1;
    current_frontier[ROOT_NODE_ID] = true;
    sol->distances[ROOT_NODE_ID] = 0;
    int exploring_distance = 0;

    while (frontier_count != 0) {

#ifdef VERBOSE
        double start_time = CycleTimer::currentSeconds();
#endif
        tracker_list_reset(next_frontier, num_nodes);
        exploring_distance += 1;
        frontier_count = bottom_up_step(graph, current_frontier, next_frontier, sol->distances, exploring_distance, 
                                        max_threads, chunk_size);
        // swap pointer
        bool * tmp = current_frontier;
        current_frontier = next_frontier;
        next_frontier = tmp;
#ifdef VERBOSE
    double end_time = CycleTimer::currentSeconds();
    printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
#endif
    }

    delete current_frontier;
    delete next_frontier;    
}

void bfs_hybrid(Graph graph, solution* sol)
{
    // CS149 students:
    //
    // You will need to implement the "hybrid" BFS here as
    // described in the handout.
}

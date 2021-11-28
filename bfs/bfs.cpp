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
// #define VERBOSE 1
#define SWITCH_THRESHOLD 25

inline void initialize_distances(
    const int num_nodes, 
    solution* sol, 
    const int max_threads,
    const int chunk_size = 64) 
{
    // initialize all nodes to NOT_VISITED
    #pragma omp parallel for num_threads (max_threads) schedule(static, chunk_size)
    for (int i = 0; i < num_nodes; i ++)
        sol->distances[i] = NOT_VISITED_MARKER;

    sol->distances[ROOT_NODE_ID] = 0;
}

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
    int* mem_offset,
    const int exploring_distance,
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
        for (int neighbor = start_edge; neighbor < end_edge; neighbor ++) {
            int outgoing = g->outgoing_edges[neighbor];

            if (distances[outgoing] == NOT_VISITED_MARKER) {
                if (__sync_bool_compare_and_swap(distances + outgoing, NOT_VISITED_MARKER, exploring_distance)) {
                    int current_index = frontier_list[thread_id].count++;
                    frontier_list[thread_id].vertices[current_index] = outgoing;
                }
            }
        }
    }
    // Collect the frontiers
    // DONE: parallelize with atomic add on total_count
    int total_count = 0;
    for (int i = 0; i < max_threads; i ++) {
        mem_offset[i] = total_count;
        total_count += frontier_list[i].count;
    }
    current_frontier.count = total_count;

    #pragma omp parallel for
    for (int i = 0; i < max_threads; i ++) {
        memcpy(current_frontier.vertices + mem_offset[i], frontier_list[i].vertices, 
                frontier_list[i].count * sizeof(int));
    }
}

// Implements top-down BFS.
//
// Result of execution is that, for each node in the graph, the
// distance to the root is stored in sol.distances.
void bfs_top_down(Graph graph, solution* sol) {

    const int max_threads {omp_get_max_threads()};
    const int vertex_set_list_size = max_threads + 1;
    
    int exploring_distance = 0;

    vertex_set* frontier_list = new vertex_set[vertex_set_list_size];   // first max_threads used for parallel processing, last one used for current_frontier
    vertex_set_list_init(frontier_list, vertex_set_list_size, graph->num_nodes);

    int* mem_offset = new int[max_threads];

    initialize_distances(graph->num_nodes, sol, max_threads);

    // Alias
    vertex_set* frontier = &(frontier_list[max_threads]);
    // setup frontier with the root node
    frontier->count = 1;
    frontier->vertices[frontier->count++] = ROOT_NODE_ID;

    while (frontier->count != 0) {

#ifdef VERBOSE
        printf("Node Count %-10d", graph->num_nodes);
        double start_time = CycleTimer::currentSeconds();
#endif
        exploring_distance += 1;
        // Clear all except the last in frontier_list
        vertex_set_list_clear(frontier_list, max_threads);
        top_down_step(graph, frontier_list, sol->distances, mem_offset, exploring_distance, max_threads);
#ifdef VERBOSE
    double end_time = CycleTimer::currentSeconds();
    printf("Step %-10d frontier=%-10d %.4f sec\n", exploring_distance, frontier->count, end_time - start_time);
#endif
    }
    delete frontier_list;
    delete mem_offset;
}

inline void tracker_list_reset(
    bool* next_frontier_tracker, 
    int list_len, 
    const int max_threads,
    const int schedule_chunk = 64) 
{
    #pragma omp parallel for num_threads (max_threads) schedule(static, schedule_chunk)
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

    
    // printf("# Threads=%-10d\n", max_threads);
    // printf("Chunk size=%-10d\n", chunk_size);
    // printf("# Nodes=%-10d\n", num_nodes);
    
    bool* current_frontier = new bool[graph->num_nodes];
    bool* next_frontier = new bool[graph->num_nodes];

    tracker_list_reset(current_frontier, num_nodes, max_threads, chunk_size);

    initialize_distances(graph->num_nodes, sol, max_threads, chunk_size / 4);

    int frontier_count = 1;
    current_frontier[ROOT_NODE_ID] = true;
    int exploring_distance = 0;

    while (frontier_count != 0) {

#ifdef VERBOSE
        double start_time = CycleTimer::currentSeconds();
#endif
        tracker_list_reset(next_frontier, num_nodes, max_threads, chunk_size);
        exploring_distance += 1;
        frontier_count = bottom_up_step(graph, current_frontier, next_frontier, sol->distances, exploring_distance, 
                                        max_threads, chunk_size);
        // swap pointer
        bool * tmp = current_frontier;
        current_frontier = next_frontier;
        next_frontier = tmp;
#ifdef VERBOSE
    double end_time = CycleTimer::currentSeconds();
    printf("Step %-10d frontier=%-10d %.4f sec\n", exploring_distance, frontier_count, end_time - start_time);
#endif
    }

    delete current_frontier;
    delete next_frontier;    
}

inline int switch_top_bottom(
    Graph g,
    vertex_set*& frontier_list,
    int* distances,
    bool* current_frontier_bool,
    const int exploring_distance,
    const int max_threads)
{
    // Run one step in top down approach

    // aliasing, frontier_list[max_threads] store the current frontier
    vertex_set& current_frontier = frontier_list[max_threads];

    int* new_node_count = new int[max_threads];
    // Write new frontier to the bool list current_frontier_bool
    #pragma omp parallel for
    for (int i = 0; i < current_frontier.count; i ++) {
        const int thread_id = omp_get_thread_num();
        int node = current_frontier.vertices[i];

        int start_edge = g->outgoing_starts[node];
        int end_edge = (node == g->num_nodes - 1)
                           ? g->num_edges
                           : g->outgoing_starts[node + 1];

        // attempt to add all neighbors to the new frontier
        for (int neighbor = start_edge; neighbor < end_edge; neighbor ++) {
            int outgoing = g->outgoing_edges[neighbor];

            if (distances[outgoing] == NOT_VISITED_MARKER) {
                if (__sync_bool_compare_and_swap(distances + outgoing, NOT_VISITED_MARKER, exploring_distance)) {
                    current_frontier_bool[outgoing] = true;
                    new_node_count[thread_id] += 1;
                }
            }
        }
    }

    int total_node_count = 0;
    for (int thread_id = 0; thread_id < max_threads; thread_id ++) {
        total_node_count += new_node_count[thread_id];
    }

    return total_node_count;
}

void bfs_hybrid(Graph graph, solution* sol)
{
    // CS149 students:
    //
    // You will need to implement the "hybrid" BFS here as
    // described in the handout.

    // Start with top down approach
    // Swith to bottom up approach if the frontier is significantly large

    bool running_top_down = true;
    bool is_switch_step = true;

    // Shared initialization
    const int max_threads {omp_get_max_threads()};
    const int num_nodes = graph->num_nodes;
    int exploring_distance = 0;
    int frontier_count = 1;

    initialize_distances(graph->num_nodes, sol, max_threads);

    // Intialize for Top Down approach
    const int vertex_set_list_size = max_threads + 1;
    
    vertex_set* frontier_list = new vertex_set[vertex_set_list_size];   // first max_threads used for parallel processing, last one used for current_frontier
    vertex_set_list_init(frontier_list, vertex_set_list_size, num_nodes);

    int* mem_offset = new int[max_threads];

    // Alias
    vertex_set* frontier = &(frontier_list[max_threads]);
    // setup frontier with the root node
    frontier->count = 1;
    frontier->vertices[frontier->count++] = ROOT_NODE_ID;

    // Initialize for Bottom Up approach
    int chunk_size = 1024;  // assuming 64 byte cache line and 1 byte bool

    while (num_nodes < max_threads * chunk_size) {
        chunk_size /= 2;
    }
    bool* current_frontier = new bool[graph->num_nodes];
    bool* next_frontier = new bool[graph->num_nodes];

    tracker_list_reset(current_frontier, num_nodes, max_threads, chunk_size);

    // Run BFS
    while (frontier_count != 0) {

#ifdef VERBOSE
        printf("Step %-10d old frontier=%-10d\n", exploring_distance, frontier_count);
        double start_time = CycleTimer::currentSeconds();
#endif
        exploring_distance += 1;
        if (running_top_down) {
        // Clear all except the last in frontier_list
            vertex_set_list_clear(frontier_list, max_threads);
            top_down_step(graph, frontier_list, sol->distances, mem_offset, exploring_distance, max_threads);
            frontier_count = frontier->count;
            if (frontier_count > 0 && num_nodes / frontier_count < SWITCH_THRESHOLD) {
                running_top_down = false;
                is_switch_step = true;
            }
        } else {
            if (is_switch_step) {
                frontier_count = switch_top_bottom(graph, frontier_list, sol->distances, current_frontier, 
                                                    exploring_distance, max_threads);
                is_switch_step = false;
            } else {
                tracker_list_reset(next_frontier, num_nodes, max_threads, chunk_size);
                frontier_count = bottom_up_step(graph, current_frontier, next_frontier, sol->distances, exploring_distance, 
                                                max_threads, chunk_size);          
                // swap pointer
                bool * tmp = current_frontier;
                current_frontier = next_frontier;
                next_frontier = tmp;          
            }

        }
#ifdef VERBOSE
        double end_time = CycleTimer::currentSeconds();
        printf("new frontier=%-10d %.4f sec ", frontier_count, end_time - start_time);
        if (running_top_down) {
            printf(" Running Top Down Approach\n");
        } else {
            if (is_switch_step) {
                printf(" Running Switch (Top -> Bottom)\n");
            } else {
                printf(" Running Bottom Up Approach\n");
            }
        }
#endif
    }

    // Clean up for Top Down approach
    delete frontier_list;
    delete mem_offset;

    // Clean up for Bottom Up approach
    delete current_frontier;
    delete next_frontier;   
}

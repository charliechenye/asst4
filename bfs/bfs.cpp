#include "bfs.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cstddef>
#include <omp.h>

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

inline void vertex_set_list_clear(vertex_set** list_pointer, const int list_size) {
    #pragma omp parallel for
    for (int i = 0 ; i < list_size; i++){
        list_pointer[i]->count = 0;
    }
}

inline void vertex_set_list_init(vertex_set** list_pointer, vertex_set* set_list, const int list_size, const int count) {
    #pragma omp parallel for
    for (int i = 0 ; i < list_size; i++){
        vertex_set_init(set_list + i, count);
        list_pointer[i] = set_list + i;
    }
}

// Take one step of "top-down" BFS.  For each vertex on the frontier,
// follow all outgoing edges, and add all neighboring vertices to the
// new_frontier.
void top_down_step(
    Graph g,
    vertex_set* frontier,
    vertex_set** frontier_list_pointer,
    int* distances,
    const int schedule_chunk = 1000)
{
    #pragma omp parallel for schedule(dynamic, schedule_chunk)
    for (int i=0; i<frontier->count; i++) {

        const int thread_id = omp_get_thread_num();
        int node = frontier->vertices[i];

        int start_edge = g->outgoing_starts[node];
        int end_edge = (node == g->num_nodes - 1)
                           ? g->num_edges
                           : g->outgoing_starts[node + 1];

        // attempt to add all neighbors to the new frontier
        for (int neighbor=start_edge; neighbor<end_edge; neighbor++) {
            int outgoing = g->outgoing_edges[neighbor];

            if (distances[outgoing] == NOT_VISITED_MARKER) {
                if (__sync_bool_compare_and_swap(distance + outgoing, NOT_VISITED_MARKER, distance[node] + 1)) {
                    int index = frontier_list_pointer[thread_id]->count++;
                    frontier_list_pointer[thread_id]->vertices[index] = outgoing;
                }
            }
        }
    }
}

// Implements top-down BFS.
//
// Result of execution is that, for each node in the graph, the
// distance to the root is stored in sol.distances.
void bfs_top_down(Graph graph, solution* sol) {

    const int max_threads {omp_get_max_threads()};
    int schedule_chunk = 100000;
    if (graph->num_nodes <= 1000)
        schedule_chunk = 500;

    vertex_set list1;
    vertex_set_init(&list1, graph->num_nodes);
    vertex_set* frontier = &list1;

    vertex_set** frontier_list_pointer = new vertex_set*[max_threads];
    vertex_set* frontier_set_list = new vertex_set[max_threads];
    vertex_set_list_init(frontier_list_pointer, frontier_set_list, max_threads, graph->num_nodes);

    // initialize all nodes to NOT_VISITED
    #pragma omp parallel for
    for (int i = 0; i < graph->num_nodes; i ++)
        sol->distances[i] = NOT_VISITED_MARKER;

    // setup frontier with the root node
    frontier->vertices[frontier->count++] = ROOT_NODE_ID;
    sol->distances[ROOT_NODE_ID] = 0;

    while (frontier->count != 0) {

#ifdef VERBOSE
        double start_time = CycleTimer::currentSeconds();
#endif

        vertex_set_list_clear(frontier_list_pointer, max_threads);

        top_down_step(graph, frontier, frontier_list_pointer, sol->distances, schedule_chunk);

#ifdef VERBOSE
    double end_time = CycleTimer::currentSeconds();
    printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
#endif

        // collect partial results into one frontier list
        // first into frontier_list_pointer[0]
        // TODO: parallel with additional variable total_count for aggregator
        for (int i = 1; i < max_threads; i ++) {
            memcpy(frontier_list_pointer[0]->vertices + frontier_list_pointer[0]->count, 
                    frontier_list_pointer[i]->vertices, 
                    frontier_list_pointer[i]->count * sizeof(int));
            frontier_list_pointer[0]->count += frontier_list_pointer[i]->count;
        }
        vertex_set* tmp = frontier;
        frontier = frontier_list_pointer[0];
        frontier_list_pointer[0] = tmp;
    }
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
}

void bfs_hybrid(Graph graph, solution* sol)
{
    // CS149 students:
    //
    // You will need to implement the "hybrid" BFS here as
    // described in the handout.
}

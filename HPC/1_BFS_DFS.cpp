#include <iostream>
#include <vector>
#include <queue>
#include <omp.h>

using namespace std;

class Graph
{
    int V;
    vector<vector<int>> adj;

public:
    Graph(int V) : V(V), adj(V) {}

    void addEdge(int v, int w)
    {
        adj[v].push_back(w);
    }

    // Parallel DFS with locking
    void parallelDFS(int startVertex)
    {
        vector<bool> visited(V, false);
        vector<omp_lock_t> locks(V);

        // Initialize locks
        for (int i = 0; i < V; ++i)
            omp_init_lock(&locks[i]);

        parallelDFSUtil(startVertex, visited, locks);

        // Destroy locks
        for (int i = 0; i < V; ++i)
            omp_destroy_lock(&locks[i]);
    }

    void parallelDFSUtil(int v, vector<bool> &visited, vector<omp_lock_t> &locks)
    {
        omp_set_lock(&locks[v]);
        if (visited[v])
        {
            omp_unset_lock(&locks[v]);
            return;
        }

        visited[v] = true;
        omp_unset_lock(&locks[v]);

        cout << v << " ";

#pragma omp parallel for shared(visited)
        for (int i = 0; i < adj[v].size(); ++i)
        {
            int n = adj[v][i];
            parallelDFSUtil(n, visited, locks); // safe because visited[n] is locked
        }
    }

    // Parallel BFS with locking
    void parallelBFS(int startVertex)
    {
        vector<bool> visited(V, false);
        queue<int> q;
        omp_lock_t queueLock;
        vector<omp_lock_t> locks(V);

        // Initialize locks
        for (int i = 0; i < V; ++i)
            omp_init_lock(&locks[i]);
        omp_init_lock(&queueLock);

        visited[startVertex] = true;
        q.push(startVertex);

        while (!q.empty())
        {
            omp_set_lock(&queueLock);
            if (q.empty())
            {
                omp_unset_lock(&queueLock);
                break;
            }

            int v = q.front();
            q.pop();
            omp_unset_lock(&queueLock);

            cout << v << " ";

#pragma omp parallel for shared(visited, q)
            for (int i = 0; i < adj[v].size(); ++i)
            {
                int n = adj[v][i];

                omp_set_lock(&locks[n]);
                if (!visited[n])
                {
                    visited[n] = true;

                    omp_set_lock(&queueLock);
                    q.push(n);
                    omp_unset_lock(&queueLock);
                }
                omp_unset_lock(&locks[n]);
            }
        }

        // Destroy locks
        for (int i = 0; i < V; ++i)
            omp_destroy_lock(&locks[i]);
        omp_destroy_lock(&queueLock);
    }
};

int main()
{
    Graph g(7);
    g.addEdge(0, 1);
    g.addEdge(0, 2);
    g.addEdge(1, 3);
    g.addEdge(1, 4);
    g.addEdge(2, 5);
    g.addEdge(2, 6);

    cout << "Depth-First Search (DFS): ";
    g.parallelDFS(0);
    cout << endl;

    cout << "Breadth-First Search (BFS): ";
    g.parallelBFS(0);
    cout << endl;

    return 0;
}

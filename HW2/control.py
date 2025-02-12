class Edge: 
    
    def __init__(self, flow, capacity, u, v):
        self.flow = flow
        self.capacity = capacity
        self.u = u
        self.v = v

# Represent a Vertex 
class Vertex:
  
    def __init__(self, h, e_flow):
        self.h = h
        self.e_flow = e_flow

        
# To represent a flow network 
class Graph:
    
    # int V;    # No. of vertices 
    # vector<Vertex> ver; 
    # vector<Edge> edge; 
    def __init__(self, V):
        
        self.V = V; 
        self.edge = []
        self.ver = []
        # all vertices are initialized with 0 height 
        # and 0 excess flow 
        for i in range(V):
            self.ver.append(Vertex(0, 0))
    
    def addEdge(self, u, v, capacity):
        # flow is initialized with 0 for all edge 
        self.edge.append(Edge(0, capacity, u, v))


    def preflow(self, s):
        
        # Making h of source Vertex equal to no. of vertices 
        # Height of other vertices is 0. 
        self.ver[s].h = len(self.ver); 

        for i in range(len(self.edge)): 
            
            # If current edge goes from source 
            if (self.edge[i].u == s):
                # Flow is equal to capacity 
                self.edge[i].flow = self.edge[i].capacity

                # Initialize excess flow for adjacent v 
                self.ver[self.edge[i].v].e_flow += self.edge[i].flow

                # Add an edge from v to s in residual graph with 
                # capacity equal to 0 
                self.edge.append(Edge(-self.edge[i].flow, 0, self.edge[i].v, s))
                

    # returns index of overflowing Vertex 
    def overFlowVertex(self):
        
        for i in range(1, len(self.ver)-1): 
            
            if(self.ver[i].e_flow > 0):
                return i

        # -1 if no overflowing Vertex 
        return -1
    

    # Update reverse flow for flow added on ith Edge 
    def updateReverseEdgeFlow(self, i, flow):
        
        u = self.edge[i].v
        v = self.edge[i].u 

        for j in range(0, len(self.edge)): 
            if (self.edge[j].v == v and self.edge[j].u == u):
                self.edge[j].flow -= flow
                return

        # adding reverse Edge in residual graph 
        e = Edge(0, flow, u, v)
        self.edge.append(e)
        

    # To push flow from overflowing vertex u 
    def push(self, u): 
        
        # Traverse through all edges to find an adjacent (of u) 
        # to which flow can be pushed 
        for i in range(0, len(self.edge)): 
            
            # Checks u of current edge is same as given 
            # overflowing vertex 
            if (self.edge[i].u == u):
                # if flow is equal to capacity then no push 
                # is possible 
                if (self.edge[i].flow == self.edge[i].capacity):
                    continue; 

                # Push is only possible if height of adjacent 
                # is smaller than height of overflowing vertex 
                if (self.ver[u].h > self.ver[self.edge[i].v].h):
                    
                    # Flow to be pushed is equal to minimum of 
                    # remaining flow on edge and excess flow. 
                    flow = min(self.edge[i].capacity - self.edge[i].flow, self.ver[u].e_flow)

                    # Reduce excess flow for overflowing vertex 
                    self.ver[u].e_flow -= flow; 

                    # Increase excess flow for adjacent 
                    self.ver[self.edge[i].v].e_flow += flow; 

                    # Add residual flow (With capacity 0 and negative 
                    # flow) 
                    self.edge[i].flow += flow; 

                    self.updateReverseEdgeFlow(i, flow); 

                    return True; 

        return False;  
    
    
    # function to relabel vertex u 
    def relabel(self, u):
        # Initialize minimum height of an adjacent 
        mh = 2100000

        # Find the adjacent with minimum height 
        for i in range(len(self.edge)):  
            if (self.edge[i].u == u):
                
                # if flow is equal to capacity then no 
                # relabeling 
                if (self.edge[i].flow == self.edge[i].capacity):
                    continue; 

                # Update minimum height 
                if (self.ver[self.edge[i].v].h < mh):
                    mh = self.ver[self.edge[i].v].h; 

                    # updating height of u 
                    self.ver[u].h = mh + 1; 

    
    # main function for printing maximum flow of graph 
    def getMaxFlow(self, s, t):
        
        self.preflow(s); 

        # loop until none of the Vertex is in overflow 
        while (self.overFlowVertex() != -1):
            
            u = self.overFlowVertex(); 
            if (self.push(u) == False):
                self.relabel(u); 

        # ver.back() returns last Vertex, whose 
        # e_flow will be final maximum flow 
        return self.ver[len(self.ver)-1].e_flow

    
# Driver program to test above functions 
V = 200; 
g = Graph(V);

with open("red_200_d.txt", 'r') as f:
    lines = f.readlines()
    for line in lines[1:]:
        parts = line.strip().split()
        if len(parts) != 3:
            raise ValueError(f"Invalid line format: {line.strip()}")
        node1, node2, weight = map(int, parts)
        g.addEdge(node1, node2, weight)

# Initialize source and sink 
s = 0
t = 199; 

#63875
print("Maximum flow is ",  g.getMaxFlow(s, t));

# The code is contributed by Arushi goel. 

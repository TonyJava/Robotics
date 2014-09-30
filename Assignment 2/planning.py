start = [0.0,0.0,0.0]
goal = [1.0,1.0,1.57]
obstacles = [(0.4,0.6),(0.4,0.4),(0.6,0.4),(0.6,0.6)]

node = [start,goal,obstacles]

#print node[2][2]
#print len(obstacles)
graph_sample=[0.0]*2
#print len(graph_sample)
#graph_sample.append((node[0][0],node[0][1]))
#print graph_sample
#print "node[2][0]: ",node[2][0]
#for i in range(2+len(obstacles)+1):
for i in range(3):
	if(i<2):
		graph_sample[i] = ((node[i][0],node[i][1]))
	elif(i>=2):
		for j in range(len(obstacles)):
#			print "j = ",j
#			print "node[2][0] : ",node[2][0]
			graph_sample.append(node[2][j])


print "graph Samples:  ",graph_sample
print "Number of samples formed : ",len(graph_sample)

'''graph_edges = [(graph_sample[0],graph_sample[1]),(graph_sample[0],graph_sample[2]),(graph_sample[0],graph_sample[3]),(graph_sample[0],graph_sample[4]),(graph_sample[0],graph_sample[5])]
'''


#function to form all possible edges from a list of samples given
def getEdges(graph_sample):
	for i in range(0,len(graph_sample)):
		for j in range(i+1,len(graph_sample)):
			graphEdge.append([graph_sample[i],graph_sample[j]])

		
graphEdge = []
getEdges(graph_sample)

print "EDGES :",graphEdge		

def getSizeOfGraphEdgeList(num):
        factorial = 1
        while (num>=1):
                factorial = factorial+num
                num = num - 1
        return factorial



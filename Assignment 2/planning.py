import math
import matplotlib.pyplot as plt


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

#to plot a list of (x,y) coordinates on graph
def plotSamplePoints(listxy):
	x = []
	y = []
	for i in range(len(listxy)):
		x.append(listxy[i][0])
	        y.append(listxy[i][1])	
	plt.scatter(x,y)
	plt.savefig('Sample Points.jpg')

#function to calculate distance between two points
def lengthOfAnEdge(edge):
	x1 = edge[0][0]
	y1 = edge[0][1]
	x2 = edge[1][0]
	y2 = edge[1][1]
	
	return round(((x2-x1)**2+(y2-y1)**2)**0.5)


#calculate the slope of an edge between (x1,y1) and (x2,y2) wrt to x-axis
def slopeOfAnEdge(edge):
	x1 = edge[0][0]
        y1 = edge[0][1]
        x2 = edge[1][0]
        y2 = edge[1][1]
	
	return math.atan((y2-y1)/(x2-x1))


#function to identify and remove all the illegal edges from the graph
def removeIllegalEdges(graphEdge,graph_sample):
	i = 0
	dynamicLengthGraphEdge = len(graphEdge)
	while (i < (dynamicLengthGraphEdge-1)):
		for j,k in zip(range(2,len(graph_sample)-2,2),range(3,len(graph_sample)-2,2)):
			if(graphEdge[i][0]==graph_sample[j] and graphEdge[i][1] == graph_sample[j+2]):
				graphEdge.remove(graphEdge[i])
				dynamicLengthGraphEdge = dynamicLengthGraphEdge - 1
			if(graphEdge[i][0]==graph_sample[k] and graphEdge[i][1] == graph_sample[k+2]):
				graphEdge.remove(graphEdge[i])
				dynamicLengthGraphEdge = dynamicLengthGraphEdge - 1
		i += 1
		
	return graphEdge		
	

#function to form all possible edges from a list of samples given
def getEdges(graph_sample):
	for i in range(0,len(graph_sample)):
		for j in range(i+1,len(graph_sample)):
			graphEdge.append([graph_sample[i],graph_sample[j]])

		
graphEdge = []
getEdges(graph_sample)

print "EDGES :",graphEdge
#print "EDGE[0][0][0] : ",graphEdge[0][0][0]

removeIllegalEdges(graphEdge,graph_sample)
print "EDGES after removing illegal ones :",graphEdge	

print "Lenght of edge ",graphEdge[1]," is ",lengthOfAnEdge(graphEdge[1])
print "Slope of edge ",graphEdge[1]," is ",slopeOfAnEdge(graphEdge[1])
plotSamplePoints(graph_sample)

def getSizeOfGraphEdgeList(num):
        factorial = 1
        while (num>=1):
                factorial = factorial+num
                num = num - 1
        return factorial



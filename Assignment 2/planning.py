import math
import matplotlib.pyplot as plt


start = [0.0,0.0,0.0]
goal = [1.0,1.0,1.57]
obstacles = [(0.4,0.6),(0.4,0.4),(0.6,0.4),(0.6,0.6)]
obstacleEdge = []
graphEdge = []

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


#Function to get edges, out of all the edges of the graph, that correspond to obstacle sides
def getEdgeAsObstacleSide(graph_sample,obstacleEdge):
	for i in range(2,len(graph_sample)):
        	for j in range(i+1,len(graph_sample)):
                	obstacleEdge.append([graph_sample[i],graph_sample[j]])


	return removeIllegalEdges(obstacleEdge,graph_sample)
	
	


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
def removeIllegalEdges(graphEdge,graph_sample):				# currently checking only for the edges
	i = 0								# that lie completely inside a polygon
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


#To check if three points are aligned in clockwise or anticlockwise orientation. Also if they are collinear.
def arePointsCCW(point1,point2,point3):
	if (((point2[0] - point1[0])*(point3[1]-point2[1]) - (point3[0]-point2[0])*(point2[1]-point1[1])) == 0):
		return False
	else:
		return (((point2[0] - point1[0])*(point3[1]-point2[1]) - (point3[0]-point2[0])*(point2[1]-point1[1])) < 0)


#rePointsCCW(line1[0],line1[1],line2[0]))function to check if two line segments are intersecting or not
def areTwoLinesIntersecting(line1,line2):
	return ((arePointsCCW(line1[0],line1[1],line2[0]))^(arePointsCCW(line1[0],line1[1],line2[1]))) and ((arePointsCCW(line2[0],line2[1] ,line1[0]))^(arePointsCCW(line2[0],line2[1],line1[1])))	
''' FOR TESTING
	print "Point 3 and 4  wrt point 1 and 2 : ",(arePointsCCW(line1[0],line1[1],line2[0]))^(arePointsCCW(line1[0],line1[1],line2[1]))
	print "Point 1 and 2 with respect to 3 and 4 : ",(arePointsCCW(line2[0],line2[1],line1[0]))^(arePointsCCW(line2[0],line2[1],line1[1]))
	print "arePointsCCW(line1[0],line1[1],line2[0]) : ",arePointsCCW(line1[0],line1[1],line2[0])
	print "arePointsCCW(line1[0],line1[1],line2[1]) : ",arePointsCCW(line1[0],line1[1],line2[1])
	print "arePointsCCW(line2[0],line2[1],line1[0]) : ",arePointsCCW(line2[0],line2[1],line1[0])
	print "arePointsCCW(line2[0],line2[1],line1[1]) : ",arePointsCCW(line2[0],line2[1],line1[1])'''
	
'''
	if (arePointsCCW(line1[0],line1[1],line2[0]) and not(arePointsCCW(line1[0],line1[1],line2[1]))) and  (not((arePointsCCW(line1[0],line1[1],line2[0])) and arePointsCCW(line1[0],line1[1],line2[1]))):
		return (2>1)
	else:
		return (1>2)
	'''
		
getEdges(graph_sample)
removeIllegalEdges(graphEdge,graph_sample)
print "Getting graph edges that are obstacle sides : ",getEdgeAsObstacleSide(graph_sample,obstacleEdge)
#print "EDGES :",graphEdge
#print "EDGE[0][0][0] : ",graphEdge[0][0][0]

#removeIllegalEdges(graphEdge,graph_sample)
#print "EDGES after removing illegal ones :",graphEdge	

#print "Lenght of edge ",graphEdge[1]," is ",lengthOfAnEdge(graphEdge[1])
#print "Slope of edge ",graphEdge[1]," is ",slopeOfAnEdge(graphEdge[1])
#plotSamplePoints(graph_sample)

'''All the tests done to test intersection of two lines...runs successfully!!!
point1 = [1,2]
point2 = [4,3]
point3 = [2,6]
point4 = [4,3]
line1 = [point1,point2]
line2 = [point3,point4]
print line1[0]
print line1[1]
print line2[0]
print line2[1]
print arePointsCCW(point1,point2,point3)
print "test : ",arePointsCCW((2,1),(6,4),(3,3))
print "Are line segments intersecting ",areTwoLinesIntersecting(line1,line2)'''

def getSizeOfGraphEdgeList(num):
        factorial = 1
        while (num>=1):
                factorial = factorial+num
                num = num - 1
        return factorial



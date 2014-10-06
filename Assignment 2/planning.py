import math
import matplotlib.pyplot as plt


start = []
goal = []
obstacles = []
obstacleEdge = []
graphEdge = []

#nodes = [start,goal,obstacles]

#print node[2][2]
#print len(obstacles)
graph_sample=[0.0]*2
#print len(graph_sample)
#graph_sample.append((node[0][0],node[0][1]))
#print graph_sample
#print "node[2][0]: ",node[2][0]
#for i in range(2+len(obstacles)+1):


#function to get the samples for the graph.
#sampling is done for visibility graph
def getGraphSamples(node):
	for i in range(3):
		if(i<2):
			graph_sample[i] = ((node[i][0],node[i][1]))
		elif(i>=2):
			for j in range(len(obstacles)):
	#			print "j = ",j
	#			print "node[2][0] : ",node[2][0]
				graph_sample.append(node[2][j])
	
	return graph_sample


#print "graph Samples:  ",graph_sample
#print "Number of samples formed : ",len(graph_sample)


'''graph_edges = [(graph_sample[0],graph_sample[1]),(graph_sample[0],graph_sample[2]),(graph_sample[0],graph_sample[3]),(graph_sample[0],graph_sample[4]),(graph_sample[0],graph_sample[5])]
'''
#Function to read the .mp file and parse it correctly to get all the coordinates, Start, Goal and Obstacles
def getTestFileInput(nameOfFile):
	file_handle = open(nameOfFile,'r')
	lines_list = file_handle.readlines()
	sample_list = []
	single_char = ""
	skipThreeStepFlag = 0
	for i in range(1, len(lines_list)-1,4):
		if(skipThreeStepFlag == 1):
			skipThreeStepFlag = 0
			i = i + 3
		for j in range((len(lines_list[i])-1)):
			if(i <= 5):
				if(lines_list[i][j]!=' '):
					single_char += lines_list[i][j]

				if(lines_list[i][j] == ' ' or lines_list[i][j+1] == '\n'):
					sample_list.append(single_char)
					single_char = ""

			if(i>5 and i < (len(lines_list)-2)):
				for m in range(i,i+4):
				#	print "m = ",m
				#       print "len(lines_list[i]) : ",len(lines_list[i])
					for n in range((len(lines_list[m])-1)):
						#print "n : ",n
						if(lines_list[m][n]!=' '):
							single_char += lines_list[m][n]
						#       print "m = ",m,"and n = ",n,"value of single_char is = ",single_char 
						if(lines_list[m][n] == ' ' or lines_list[m][n+1] == '\n'):
							sample_list.append(single_char)
							single_char = ""

				skipThreeStepFlag = 1
				break

		if(i == (len(lines_list) - (len(lines_list)%4))):
	                break


	return sample_list


#function to extract Start, Goal, and Obstacle coordinates from the list of coorinates received from the parser
def getAllCoordinates(sample_list):
	
	nodes = []
	skipNextIterationFlag = 0			# to skip one iteration so that coordinates of obstacles 
							# can be shown as combinations of (x,y)
	for i in range(len(sample_list)):
		if(skipNextIterationFlag == 1):
			skipNextIterationFlag = 0
			continue
		if(i<=2):
			start.append(float(sample_list[i]))
		elif(i>2 and i <=5):
			goal.append(float(sample_list[i]))
		elif(i>5):
			obstacles.append((float(sample_list[i]),float(sample_list[i+1])))	
			skipNextIterationFlag = 1
	
	nodes = [start,goal,obstacles]
	return nodes


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


#function to check if two line segments are intersecting or not
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
		
removeIllegalEdges(graphEdge,graph_sample)
print "Getting graph edges that are obstacle sides : ",getEdgeAsObstacleSide(graph_sample,obstacleEdge)




'''First function call reads the .mp file and returns data in the form ['0.0','0.0'...] where all the coordinates
   are listed individually. Second function call retrieves all the coordinates from above list in following fomat:
   [(start),(goal),(all obstacles)].
   Third functionc all gives the list of all the samples as visibility graph '''

print getGraphSamples(getAllCoordinates(getTestFileInput('test3.mp')))  


plotSamplePoints(graph_sample)
getEdges(graph_sample)
removeIllegalEdges(graphEdge,graph_sample)
print "****GRAPH EDGES****"
print graphEdge
print "Number of EDGES FORMED: ",len(graphEdge)
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



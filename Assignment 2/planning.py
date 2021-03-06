import sys
import math
import matplotlib.pyplot as plt
from matplotlib import collections as mc
import pylab as pl


start = []
goal = []
obstacles = []
obstacleEdge = []
graphEdge = []
finalPath = []

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
	
	graph_sample.append(graph_sample[1])
	graph_sample.remove(graph_sample[1])
	return graph_sample


#print "graph Samples:  ",graph_sample
#print "Number of samples formed : ",len(graph_sample)



'''graph_edges = [(graph_sample[0],graph_sample[1]),(graph_sample[0],graph_sample[2]),(graph_sample[0],graph_sample[3]),(graph_sample[0],graph_sample[4]),(graph_sample[0],graph_sample[5])]
'''
#Function to read the .mp file and parse it correctly to get all the coordinates, Start, Goal and Obstacles
def getTestFileInput(nameOfFile):
	file_handle = open(nameOfFile,'r')
	lines_list = file_handle.readlines()
	print "Lines list read by the readlines func : ", lines_list
	#print "\nlines_list[9] = ",lines_list[9]
	sample_list = []
	single_char = ""
	skipThreeStepFlag = 0
	if(nameOfFile != 'test4.mp'):
		for i in range(1, len(lines_list)-1,4):
			if(skipThreeStepFlag == 1 and len(lines_list)>15):
				skipThreeStepFlag = 0
				i = i + 3
			print "\nValue of I : ",i
			if(i>len(lines_list)-1):
				break
			print "\nlen(lines_list[",i,"] : ",(lines_list[i])

			for j in range((len(lines_list[i])-1)):   		# Replacing (len(lines_list[i])-1) with (len(lines_list[i]))
				print "\nValue of J with each iteration : ",j
				if(i <= 5):
					if(lines_list[i][j]!=' '):
						single_char += lines_list[i][j]
					
			
					if(lines_list[i][j] == ' ' or lines_list[i][j+1] == '\n'):
	#					print "Single_char : ",single_char
						sample_list.append(single_char)
						single_char = ""
					
	#				print "(len(lines_list[",i,"]))",len(lines_list[i])
	#			print "sample_list : ",sample_list	
				if(i>5 and i < (len(lines_list)-2)):		#Replacing len(lines_list)-2) with len(lines_list))
					for m in range(i,i+4):
	#					print "m = ",m
					#       print "len(lines_list[i]) : ",len(lines_list[i])
						for n in range((len(lines_list[m])-1)):
	#						print "n : ",n
							if(lines_list[m][n]!=' '):
								single_char += lines_list[m][n]
							#       print "m = ",m,"and n = ",n,"value of single_char is = ",single_char 
							if(lines_list[m][n] == ' ' or lines_list[m][n+1] == '\n'):
	#							print "\nSingle char when checking for obstacle : ",single_char
							#	if(single_char != '<\\Obstacles>'):
								sample_list.append(single_char)
								single_char = ""

					skipThreeStepFlag = 1
					break
			
			if(i == (len(lines_list) - (len(lines_list)%4))):
			        break
	
	if(nameOfFile == 'test4.mp' or nameOfFile == 'test5.mp'):
		#print "\nlines list when reading 4th input : ",lines_list
				
		for i in range(1,6,4):
			for j in range((len(lines_list[i])-1)):
				if(lines_list[i][j]!=' '):
						single_char += lines_list[i][j]
					
				if(lines_list[i][j] == ' ' or lines_list[i][j+1] == '\n'):
	#					print "Single_char : ",single_char
						sample_list.append(single_char)
						single_char = ""
			
		if(nameOfFile == 'test4.mp'):
			maxIndex = 51
		elif(nameOfFile == 'test5.mp'):
			maxIndex = 58
		for i in range(9,maxIndex,7):
#			print "Inside first for\n"
			for j in range(i,i+4):
				#print "Inside second for\n"
				for m in range(0,len(lines_list[j])-1):
					#print "Inside third for\n"
					#print "i = ",i,"  j = ",j,"  m =  ",m
  					if(lines_list[j][m]!=' '):
						single_char += lines_list[j][m]
					#       print "m = ",m,"and n = ",n,"value of single_char is = ",single_char 
					if(lines_list[j][m] == ' ' or lines_list[j][m+1] == '\n'):
						#print "\nSingle char when checking for obstacle : ",single_char
					#	if(single_char != '<\\Obstacles>'):
						sample_list.append(single_char)
						single_char = ""
		print "sample list when test4.mp is read : ",sample_list			

	return sample_list


#function to extract Start, Goal, and Obstacle coordinates from the list of coorinates received from the parser
def getAllCoordinates(sample_list):
	
	nodes = []
	skipNextIterationFlag = 0			# to skip one iteration so that coordinates of obstacles 
	print "\nsample list : ",sample_list		# can be shown as combinations of (x,y)
	for i in range(len(sample_list)):
		if(skipNextIterationFlag == 1):
			skipNextIterationFlag = 0
			continue
		if(i<=2):
			start.append(float(sample_list[i]))
			print "Start : ",start
		elif(i>2 and i <=5):
			goal.append(float(sample_list[i]))
			print "Goal : ",goal
		elif(i>5):
			print "i : ",i
			print "sample_list[",i,"] : ",sample_list[i]
			print "sample_list[",i+1,"] : ",sample_list[i+1]
			obstacles.append((float(sample_list[i]),float(sample_list[i+1])))	
			skipNextIterationFlag = 1
	
	nodes = [start,goal,obstacles]
	return nodes


#Function to get edges, out of all the edges of the graph, that correspond to obstacle sides
def getEdgeAsObstacleSide(graph_sample,obstacleEdge):
	obstacleFirstVertex = 0.0
	j = 0
	for i in range(1,len(graph_sample)-1,4):
		obstacleFirstVertex = i
	    	for j in range(i,i+3):
                	obstacleEdge.append([graph_sample[j],graph_sample[j+1]])
		obstacleEdge.append([graph_sample[obstacleFirstVertex],graph_sample[j+1]])

	return obstacleEdge
	
	


#to plot a list of (x,y) coordinates on graph
def plotSamplePoints(listxy):
	x = []
	y = []
	for i in range(len(listxy)):
		x.append(listxy[i][0])
	        y.append(listxy[i][1])	
	plt.scatter(x,y)
	plt.plot(x,y)
	plt.savefig('Sample Points.jpg')


#function to plot line segments on graph
def plotGraphEdges(graphEdge):
#	lines = [[(0, 1), (1, 1)], [(2, 3), (3, 3)], [(1, 2), (1, 3)]]
#	c = np.array([(1, 0, 0, 1), (0, 1, 0, 1), (0, 0, 1, 1)])

	lc = mc.LineCollection(graphEdge, linewidths=2)
	fig, ax = pl.subplots()
	ax.add_collection(lc)
	ax.autoscale()
	ax.margins(0.1)
	pl.savefig('Sample Points.jpg')
	

#function to calculate distance between two points
def lengthOfAnEdge(edge):
	x1 = edge[0][0]
	y1 = edge[0][1]
	x2 = edge[1][0]
	y2 = edge[1][1]
	
	return round(((x2-x1)**2+(y2-y1)**2)**0.5,2)


#calculate the slope of an edge between (x1,y1) and (x2,y2) wrt to x-axis
def slopeOfAnEdge(edge):
	x1 = edge[0][0]
        y1 = edge[0][1]
        x2 = edge[1][0]
        y2 = edge[1][1]
	
	return math.atan((y2-y1)/(x2-x1))


#function to identify and remove all the illegal edges from the graph
def removeIllegalEdgesObstacleDiagonal(graphEdge,graph_sample):		# checking only for the edges
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
	


#Function to check if an edge coming from outside an obstacle is passing through the obstacle or not.
def removeIllegalEdgesInsideObstacle(graphEdge,obstacleEdge):
	i = 0
	countOfEdgesRemoved = 0
	lengthOfGraphEdge = len(graphEdge)
	listOfIndexToBeRemoved = []
	while(i<lengthOfGraphEdge):
		for j in range(len(obstacleEdge)):
#			print "i = ",i,"    and j = ",j
			if(areTwoLinesIntersecting(graphEdge[i],obstacleEdge[j])):
#				print "Edges checked for interesection : ",graphEdge[i],obstacleEdge[j]
#				print "Are intersecting ? : ",areTwoLinesIntersecting(graphEdge[i],obstacleEdge[j])
#				print "Edge that will be removed : ",graphEdge[i]
				listOfIndexToBeRemoved.append(i)
#				lengthOfGraphEdge -= 1
				#print "Graph edge removed is : ",graphEdge[i]
				break
#				graphEdge.remove(graphEdge[i])
		i += 1
	print "list of the index to be removed : ",listOfIndexToBeRemoved	
	for k in range(len(listOfIndexToBeRemoved)):
		
		graphEdge.remove(graphEdge[listOfIndexToBeRemoved[k]-countOfEdgesRemoved])
		countOfEdgesRemoved += 1
	
	print "countOfEdgesRemoved :",countOfEdgesRemoved
	return graphEdge			

#function to form all possible edges from a list of samples given
def getEdges(graph_sample):
	for i in range(0,len(graph_sample)):
		for j in range(i+1,len(graph_sample)):
			graphEdge.append([graph_sample[i],graph_sample[j]])


#To check if three points are aligned in clockwise or anticlockwise orientation. Also if they are collinear.
def arePointsCCW(point1,point2,point3):
#	if (((point2[0] - point1[0])*(point3[1]-point2[1]) - (point3[0]-point2[0])*(point2[1]-point1[1])) == 0):
#		return False
#	else:
	return ((point3[1] - point1[1])*(point2[0]-point1[0]) > (point2[1]-point1[1])*(point3[0]-point1[0]))


#function to check if two line segments are intersecting or not
def areTwoLinesIntersecting(line1,line2):
	'''FOR TESTING
	print "Point 3 and 4  wrt point 1 and 2 : ",(arePointsCCW(line1[0],line1[1],line2[0]))^(arePointsCCW(line1[0],line1[1],line2[1]))
	print "Point 1 and 2 with respect to 3 and 4 : ",(arePointsCCW(line2[0],line2[1],line1[0]))^(arePointsCCW(line2[0],line2[1],line1[1]))
	print "arePointsCCW(line1[0],line1[1],line2[0]) : ",arePointsCCW(line1[0],line1[1],line2[0])
	print "arePointsCCW(line1[0],line1[1],line2[1]) : ",arePointsCCW(line1[0],line1[1],line2[1])
	print "arePointsCCW(line2[0],line2[1],line1[0]) : ",arePointsCCW(line2[0],line2[1],line1[0])
	print "arePointsCCW(line2[0],line2[1],line1[1]) : ",arePointsCCW(line2[0],line2[1],line1[1])'''
#	return ((arePointsCCW(line1[0],line1[1],line2[0]))^(arePointsCCW(line1[0],line1[1],line2[1]))) and ((arePointsCCW(line2[0],line2[1] ,line1[0]))^(arePointsCCW(line2[0],line2[1],line1[1])))
	
#	return (arePointsCCW(line1[0],line2[0],line2[1]) != arePointsCCW(line1[1],line2[0],line2[1])) and (arePointsCCW(line1[0],line1[1],line2[0]) != arePointsCCW(line1[0],line1[1],line2[1]))
	if((arePointsCCW(line1[0],line2[0],line2[1]) != arePointsCCW(line1[1],line2[0],line2[1])) and (arePointsCCW(line1[0],line1[1],line2[0]) != arePointsCCW(line1[0],line1[1],line2[1]))):
		if(line1[0] == line2[0] or line1[1] == line2[0] or line1[0] == line2[1] or line1[1] == line2[1]):
			return False
		else:
			 return True	
	else:
		return False
#if (arePointsCCW(line1[0],line1[1],line2[0]) and not(arePointsCCW(line1[0],line1[1],line2[1]))) and  (not((arePointsCCW(line1[0],line1[1],line2[0])) and arePointsCCW(line1[0],line1[1],line2[1]))):
#		return (2>1)
#	else:
#		return (1>2)



removeIllegalEdgesObstacleDiagonal(graphEdge,graph_sample)

'''First function call reads the .mp file and returns data in the form ['0.0','0.0'...] where all the coordinates
   are listed individually. Second function call retrieves all the coordinates from above list in following fomat:
   [(start),(goal),(all obstacles)].
   Third functionc gives the list of all the samples as visibility graph '''


arg1 = sys.argv[1]
#Remove the comment after test
#sample_list = getTestFileInput(arg1)
#print "\n Sample List : ",sample_list
print getGraphSamples(getAllCoordinates(getTestFileInput(arg1)))

plotSamplePoints(graph_sample)
getEdges(graph_sample)
plotSamplePoints(graph_sample)
removeIllegalEdgesObstacleDiagonal(graphEdge,graph_sample)
print "****GRAPH EDGES****"
print graphEdge
print "Number of EDGES FORMED: ",len(graphEdge)
print "****OBSTACLE EDGES*****"
print getEdgeAsObstacleSide(graph_sample,obstacleEdge)
print "Graph edges after removing second category of illegal edges : "
removeIllegalEdgesInsideObstacle(graphEdge,obstacleEdge)
print graphEdge
print "Number of EDGES REMAINED :",len(graphEdge)

plotGraphEdges(graphEdge)


#print "EDGES :",graphEdge
#print "EDGE[0][0][0] : ",graphEdge[0][0][0]

#removeIllegalEdges(graphEdge,graph_sample)
#print "EDGES after removing illegal ones :",graphEdge	

#print "Lenght of edge ",graphEdge[1]," is ",lengthOfAnEdge(graphEdge[1])
#print "Slope of edge ",graphEdge[1]," is ",slopeOfAnEdge(graphEdge[1])
#plotSamplePoints(graph_sample)

#All the tests done to test intersection of two lines...runs successfully!!!
#Cases where one end point of one of the line segments lies on the other 
#segment, they are not considered to be in collision

'''
point1 = [0,0]
point2 = [5,5]
point3 = [5,5]
point4 = [5,3]
line1 = [point1,point2]
line2 = [point3,point4]
print line1[0]
print line1[1]
print line2[0]
print line2[1]
print arePointsCCW(point1,point2,point3)
#print "test : ",arePointsCCW((2,1),(6,4),(3,3))
print "Are line segments intersecting ",areTwoLinesIntersecting(line1,line2)
'''

## Applying DIJKSTRA's ALGORITHM ##

graph_sampleDistancePrevious = []


#function to find out node with minimum distance value from graph_sampleDistancePrevious list
def minDistanceSampleIndex(graph_sampleDistancePrevious):
	minIndex = 0
	minDistance = 0.0
	for i in range(len(graph_sampleDistancePrevious)):
		if(minDistance >= graph_sampleDistancePrevious[i][1]):
			minDistance = graph_sampleDistancePrevious[i][1]
			minIndex = i

	return minIndex

#function to return a list of neighbouring vertices for a point
def getNeighbourOfASample(samplePoint,neighbourVertex):
#	print "Getting neighbour for : ",samplePoint
	for i in range(len(graphEdge)):
		if(graphEdge[i][0] == samplePoint):
			neighbourVertex.append(graphEdge[i][1])
		elif(graphEdge[i][1] == samplePoint):
			neighbourVertex.append(graphEdge[i][0])

	return neighbourVertex
	

#function to return index of a (x,y) point from a list
def getIndexOfAPointFromPredGraph(point,graph_sampleDistancePrevious):
	
	for i in range(len(graph_sampleDistancePrevious)):
		if(graph_sampleDistancePrevious[i][0] == point):
			return i



#function to check if an element exists in a list
def doesElementExist(element,targetList):
	for i in range(len(targetList)):
		if(targetList[i] == element):
			return True
		else:
			return False


#function to get path from informed list generated by Dijkstra's algorithm
def getPathFromDijkstraList(graph_sampleDistancePrevious,node,start):
	global finalPath
	for i in range(len(graph_sampleDistancePrevious)):
		if(graph_sampleDistancePrevious[i][0] == node):
			print "\nInside first if"
			print "Value at graph_sampleDistancePrevious[",i,"][0] is ",graph_sampleDistancePrevious[i][0]
			finalPath.append(graph_sampleDistancePrevious[i][0])

			if(graph_sampleDistancePrevious[i][2] == start):
 				print "\nInside second level if"
				finalPath.append(graph_sampleDistancePrevious[i][2])
				print "finalPath : ",finalPath
				return
			if(graph_sampleDistancePrevious[i][2] != start):
				getPathFromDijkstraList(graph_sampleDistancePrevious,graph_sampleDistancePrevious[i][2],start)
			
		

#function to reverse the path in final list
def reversingPath():
	global finalPath
	temp = []
	i = len(finalPath)-1
	while(i>=0):
		temp.append(finalPath[i])
		i -= 1
	finalPath = temp
	return finalPath



#Dijktra's algorithm
def dijkstra(graph_sample,graphEdge,start,goal):
	unvisitedSample = []
	print "len(graph_sample) : ",len(graph_sample)
	for i in range(len(graph_sample)):		#len(graph_sample)
		if(i == 0):
			temp = [(graph_sample[i]),0.0,(9.9,9.9)]
			graph_sampleDistancePrevious.append(temp)
			unvisitedSample.append(graph_sample[i])

		if(i > 0):
			temp = [(graph_sample[i]),999.9,(9.9,9.9)]
			graph_sampleDistancePrevious.append(temp)
			unvisitedSample.append(graph_sample[i])
		#graph_sampleDistancePrevious[i][0] = graph_sample[i]
		#graph_sampleDistancePrevious[i][1] = 999.9
		#graph_sampleDistancePrevious[i][2] = (9.9,9.9)
	
#	print "Printing graph_sampleDistancePrevious[0][1] from inside Dijkstra's: ",graph_sampleDistancePrevious[0][1]		
	k = 0	
	print "Before entering while loop***************"
	print "graph with predessor and univisted lists ******************"
	print graph_sampleDistancePrevious
	print unvisitedSample
	while(k<len(unvisitedSample)):
		minDistanceIndex = minDistanceSampleIndex(graph_sampleDistancePrevious) 		
#		print "Unvisited graph: ",unvisitedSample
			
	#	print "minDistanceIndex  : ",minDistanceIndex
#		print "graph_sampleDistancePrevious : ",graph_sampleDistancePrevious
	#	print "graph_sampleDistancePrevious[minDistanceIndex]: ",graph_sampleDistancePrevious[minDistanceIndex]
	#	print "graph_sampleDistancePrevious[minDistanceIndex][0] : ",graph_sampleDistancePrevious[minDistanceIndex][0]
		if(doesElementExist(graph_sampleDistancePrevious[minDistanceIndex][0],unvisitedSample)):
			unvisitedSample.remove(graph_sampleDistancePrevious[minDistanceIndex][0])
		neighbourVertex = []
		neighbourVertex = getNeighbourOfASample(graph_sampleDistancePrevious[k][0],neighbourVertex)
#		print "neighbour Vertex for k =  ",k,"is : ",neighbourVertex
		for i in range(len(neighbourVertex)):
			#print " graph_sampleDistancePrevious[k][0]  ",graph_sampleDistancePrevious[k][0]
			#print " neighbourVertex[i] ",neighbourVertex[i]
			#print "lengthOfAnEdge([graph_sampleDistancePrevious[k][0],neighbourVertex[i]]) : ",lengthOfAnEdge([graph_sampleDistancePrevious[k][0],neighbourVertex[i]])
			tempDistance = graph_sampleDistancePrevious[k][1] + lengthOfAnEdge([graph_sampleDistancePrevious[k][0],neighbourVertex[i]])			
			vertexIndexInGraph = getIndexOfAPointFromPredGraph(neighbourVertex[i],graph_sampleDistancePrevious)
			if(tempDistance < graph_sampleDistancePrevious[vertexIndexInGraph][1]):
	#			print graph_sampleDistancePrevious
#				print "graph_sampleDistancePrevious[vertexIndexInGraph][1] ", graph_sampleDistancePrevious[vertexIndexInGraph][1]
#				print " tempDistance : ",tempDistance
#				print " vertexIndexInGraph : ",vertexIndexInGraph
				graph_sampleDistancePrevious[vertexIndexInGraph][1] = tempDistance
				graph_sampleDistancePrevious[vertexIndexInGraph][2] = graph_sampleDistancePrevious[k][0] 
			
			if(graph_sampleDistancePrevious[vertexIndexInGraph][2]==goal):
				return graph_sampleDistancePrevious
		k += 1

		
#print "minDistanceSampleIndex(graph_sampleDistancePrevious) ",minDistanceSampleIndex(graph_sampleDistancePrevious)	
print dijkstra(graph_sample,graphEdge,start,goal)
print "Finale graph_sampleDistancePrevious list : ",graph_sampleDistancePrevious
goalCoordinates = (goal[0],goal[1])
startCoordinates = (start[0],start[1])

print "\n Goal is: ",goalCoordinates
print "\n Start is: ",startCoordinates
path = []
print "\nFinal path after Dijktras is : " 
getPathFromDijkstraList(graph_sampleDistancePrevious,goalCoordinates,startCoordinates)
reversingPath()
print finalPath
 


##Identifying the path with turns (arcs)


finalPathWithOrientation = []
#functiont to get the list of orientation for path points
def getOrientationForPathPoints():
	global finalPath
	global finalPathWithOrientation
	global start
	global goal
	for i in range(len(finalPath)):
		if(i==0):
			finalPathWithOrientation.append([finalPath[i],start[2]])
		elif(i==len(finalPath)-1):
			finalPathWithOrientation.append([finalPath[i],goal[2]])
	
		else:
			finalPathWithOrientation.append([finalPath[i],99.99])
	return finalPathWithOrientation


pathWithArcPoints = []
pathLength = 0.0

#function to get path with arc points
def getPathWithArc():
	global finalPathWithOrientation
	global pathWithArcPoints
	global goal	
	global pathLength	
	for i in range(len(finalPathWithOrientation)-1):
		pathWithArcPoints.append(finalPathWithOrientation[i])
		
		#finding end point for first arc
		theta_1 = finalPathWithOrientation[i][1]
		print "\ntheta_1 : ",theta_1
		edge = (finalPathWithOrientation[i][0],finalPathWithOrientation[i+1][0])
		theta_2 = slopeOfAnEdge(edge)
		if(theta_2-theta_1<0):
			delta_theta = -1* (theta_2-theta_1)
		else:
			delta_theta = theta_2 - theta_1
		print "\ndelta_theta 1: ",delta_theta
		r = (0.1/(2**0.5))
		x1 = finalPathWithOrientation[i][0][0] + r*math.sin(delta_theta/2)*math.cos(theta_1+ delta_theta/2)
		print "\nx1 : ",x1
		print "\nfinalPathWithOrientation[i][0][0] : ",finalPathWithOrientation[i][0][0]
		y1 = finalPathWithOrientation[i][0][1] + r*math.sin(delta_theta/2)*math.sin(theta_1+ delta_theta/2)
		print "\ny1 : ",y1
		orientation = theta_2
		pathWithArcPoints.append([(x1,y1),orientation])
		arcLength1 = r*delta_theta

		#finding the start point of the second arc since end point is known - the i+1 point
		edge = (finalPathWithOrientation[i][0],finalPathWithOrientation[i+1][0])
		theta_1 = slopeOfAnEdge(edge)
		print " \n finalPathWithOrientation[i+1][0] : ",finalPathWithOrientation[i+1][0]
		print " \ngoal[0] : ",(goal[0],goal[1]) 
		if(finalPathWithOrientation[i+1][0] == (goal[0],goal[1])):
			print "\nTheta 2 acquired as : ",finalPathWithOrientation[i+1][1]
			theta_2 = finalPathWithOrientation[i+1][1]
		else:
			edge = (finalPathWithOrientation[i][0],finalPathWithOrientation[i+1][0])
			theta_2 = 2*slopeOfAnEdge(edge)
			
	
		delta_theta = theta_2 - theta_1
		print "\ndelta_theta 2: ",delta_theta
		r = (0.1/(2**0.5))
		x2 = finalPathWithOrientation[i+1][0][0] - r*math.sin(delta_theta/2)*math.cos(theta_1+ delta_theta/2)
		y2 = finalPathWithOrientation[i+1][0][1] - r*math.sin(delta_theta/2)*math.sin(theta_1+ delta_theta/2)
		orientation = theta_2/2
		pathWithArcPoints.append([(x2,y2),orientation])
		arcLength2 = r*delta_theta

		pathLength = arcLength1 + arcLength2 + lengthOfAnEdge(((x1,y1),(x2,y2)))
		pathWithArcPoints.append([finalPathWithOrientation[i+1][0],theta_2])
		
		
		
	return 		
	

print "\n****************************\n"
print "Making final path list with orientation : ",getOrientationForPathPoints()
print "\n****************************\n"
getPathWithArc()
print "Path with arc points : ", pathWithArcPoints
print "\nLength of the arc : ",pathLength

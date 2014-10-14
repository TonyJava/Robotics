import sys
import math
import matplotlib.pyplot as plt
from matplotlib import collections as mc
import pylab as pl
import copy

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
	sample_list = []
	single_char = ""
	skipThreeStepFlag = 0
	if(nameOfFile != 'test4.mp' and nameOfFile!='test5.mp'):
		for i in range(1, len(lines_list)-1,4):
			if(skipThreeStepFlag == 1 and len(lines_list)>15):
				skipThreeStepFlag = 0
				i = i + 3
			if(i>len(lines_list)-1):
				break

			for j in range((len(lines_list[i])-1)):  
				if(i <= 5):
					if(lines_list[i][j]!=' '):
						single_char += lines_list[i][j]
					
			
					if(lines_list[i][j] == ' ' or lines_list[i][j+1] == '\n'):
	
						sample_list.append(single_char)
						single_char = ""
					
	
	
				if(i>5 and i < (len(lines_list)-2)):		
					for m in range(i,i+4):
						for n in range((len(lines_list[m])-1)):
							if(lines_list[m][n]!=' '):
								single_char += lines_list[m][n]
							if(lines_list[m][n] == ' ' or lines_list[m][n+1] == '\n'):
								sample_list.append(single_char)
								single_char = ""

					skipThreeStepFlag = 1
					break
			
			if(i == (len(lines_list) - (len(lines_list)%4))):
			        break
	
	if(nameOfFile == 'test4.mp' or nameOfFile == 'test5.mp'):
						
		for i in range(1,6,4):
			for j in range((len(lines_list[i])-1)):
				if(lines_list[i][j]!=' '):
						single_char += lines_list[i][j]
					
				if(lines_list[i][j] == ' ' or lines_list[i][j+1] == '\n'):
						sample_list.append(single_char)
						single_char = ""
		maxIndex = 0	
		if(nameOfFile == 'test4.mp'):
			maxIndex = 58
		elif(nameOfFile == 'test5.mp'):
			print "\n Test 5 identified"
			maxIndex = 65
		for i in range(9,maxIndex,7):
			for j in range(i,i+4):
				for m in range(0,len(lines_list[j])-1):
					if(lines_list[j][m]!=' '):
						single_char += lines_list[j][m]
					
					if(lines_list[j][m] == ' ' or lines_list[j][m+1] == '\n'):
						sample_list.append(single_char)
						single_char = ""


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

		elif(i>2 and i <=5):
			goal.append(float(sample_list[i]))

		elif(i>5):
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
	#plt.plot(x,y)
	#plt.show()
	plt.savefig('Sample Points.jpg')


#function to plot line segments on graph
def plotGraphEdges(graphEdge):
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
	if((x2-x1)== 0):
		return 0.0
	else:	
		return math.atan((y2-y1)/(x2-x1))


#function to identify and remove all the illegal edges from the graph
def removeIllegalEdgesObstacleDiagonal(graphEdge,graph_sample):		# checking only for the edges
	i = 0															# that lie completely inside a polygon

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
			if(areTwoLinesIntersecting(graphEdge[i],obstacleEdge[j])):
				listOfIndexToBeRemoved.append(i)
				break

		i += 1

	for k in range(len(listOfIndexToBeRemoved)):
		
		graphEdge.remove(graphEdge[listOfIndexToBeRemoved[k]-countOfEdgesRemoved])
		countOfEdgesRemoved += 1
	
	return graphEdge			


#function to remove edges overlapping with the environment boundaries
def removeEdgesOverlappingBoundary(graphEdge):
	maxIndex = len(graphEdge)
	countOfRemovals = 0
	listToBeRemoved = []
	for i in range(maxIndex):
		if((graphEdge[i][0][0] == 0.0 and graphEdge[i][1][0] == 0.0) or (graphEdge[i][0][1] == 0.0 and graphEdge[i][1][1] == 0.0)):
			
			listToBeRemoved.append(i)
		elif((graphEdge[i][0][0] == 1.0 and graphEdge[i][1][0] == 1.0) or (graphEdge[i][0][1] == 1.0 and graphEdge[i][1][1] == 1.0)):
			listToBeRemoved.append(i)

	j=len(listToBeRemoved)-1
	while(j>=0):
		graphEdge.remove(graphEdge[listToBeRemoved[j]])

		j -= 1

	return graphEdge
				
		
#function to form all possible edges from a list of samples given
def getEdges(graph_sample):
	for i in range(0,len(graph_sample)):
		for j in range(i+1,len(graph_sample)):
			graphEdge.append([graph_sample[i],graph_sample[j]])


#To check if three points are aligned in clockwise or anticlockwise orientation. Also if they are collinear.
def arePointsCCW(point1,point2,point3):
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
	if((arePointsCCW(line1[0],line2[0],line2[1]) != arePointsCCW(line1[1],line2[0],line2[1])) and (arePointsCCW(line1[0],line1[1],line2[0]) != arePointsCCW(line1[0],line1[1],line2[1]))):
		if(line1[0] == line2[0] or line1[1] == line2[0] or line1[0] == line2[1] or line1[1] == line2[1]):
			return False
		else:
			 return True	
	else:
		return False



removeIllegalEdgesObstacleDiagonal(graphEdge,graph_sample)

'''First function call reads the .mp file and returns data in the form ['0.0','0.0'...] where all the coordinates
   are listed individually. Second function call retrieves all the coordinates from above list in following fomat:
   [(start),(goal),(all obstacles)].
   Third functionc gives the list of all the samples as visibility graph '''


arg1 = sys.argv[1]
#Remove the comment after test
#sample_list = getTestFileInput(arg1)
#print "\n Sample List : ",sample_list
getGraphSamples(getAllCoordinates(getTestFileInput(arg1)))

plotSamplePoints(graph_sample)
getEdges(graph_sample)
print "\n\n!!!!!!!!!!!!! printing RAW EDGES!!!!!!!!!!!!!!!!!!!1\n",graphEdge
removeIllegalEdgesObstacleDiagonal(graphEdge,graph_sample)
#print "****GRAPH EDGES****"
print graphEdge
#print "Number of EDGES FORMED: ",len(graphEdge)
#print "****OBSTACLE EDGES*****"
print getEdgeAsObstacleSide(graph_sample,obstacleEdge)
#print "\nGraph edges after removing second category of illegal edges : "
removeIllegalEdgesInsideObstacle(graphEdge,obstacleEdge)
print "\nGraph before removing boundary edges !!!!!!!!!!!!:\n",graphEdge
print "\nGraph after removing boundary edges !!!!!!!!!!!!:\n ",removeEdgesOverlappingBoundary(graphEdge)




#All the tests done to test intersection of two lines...runs successfully!!!
#Cases where one end point of one of the line segments lies on the other 
#segment, they are not considered to be in collision


'''
point1 = [0.2,0.7]
point2 = [1.0,1.0]
point3 = [0.2,0.85]
point4 = [0.8,0.85]
line1 = [point1,point2]
line2 = [point3,point4]
print line1[0]
print line1[1]
print line2[0]
print line2[1]
print arePointsCCW(point1,point2,point3)
#print "test : ",arePointsCCW((2,1),(6,4),(3,3))
print "\n********************************Are line segments intersecting *******************************\n",areTwoLinesIntersecting(line1,line2)
'''

## Applying DIJKSTRA's ALGORITHM ##

graph_sampleDistancePrevious = []


#function to find out node with minimum distance value from graph_sampleDistancePrevious list
def minDistanceSampleIndex(unvisitedListwDistance):
#	print "\nMinimum Index func called"	
	minIndex = 0
	minDistance = 0.0
	listOfDistance = []
	#print "\n\n *** Inside minimum distance calculation , unvisitedListwDistance: ", unvisitedListwDistance
	for i in range(len(unvisitedListwDistance)):
		listOfDistance.append(unvisitedListwDistance[i][1])
	minIndex = listOfDistance.index(min(listOfDistance)) 
#		if(minDistance >= sampleListCopy[i][1]):
#			minDistance = sampleListCopy[i][1]
#			minIndex = i

	return minIndex,unvisitedListwDistance[minIndex][0]


#function to return a list of neighbouring vertices for a point
def getNeighbourOfASample(samplePoint,neighbourVertex):
	global graphEdge
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
			finalPath.append(graph_sampleDistancePrevious[i][0])

			if(graph_sampleDistancePrevious[i][2] == start):
				finalPath.append(graph_sampleDistancePrevious[i][2])

				return
			if(graph_sampleDistancePrevious[i][2] != start):
				if(graph_sampleDistancePrevious[i][2] == (0.85,0.1) and graph_sampleDistancePrevious[i][0] == (0.85,0.15)): #and graph_sampleDistancePrevious[i][2][1] == 0.1):
					print "\nPOINT FOUND\n "
				print "\n Point checked currently : ",graph_sampleDistancePrevious[i][0]
				print "\nPredessor for point checked currently : ",graph_sampleDistancePrevious[i][2]
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

goalCoordinates = (goal[0],goal[1])


#function to attach distances to unvisited sample list
def getUnvisitedSamplewDistance(unvisitedList,graphDistPred):
	unvisitedListwDistance = []
	for i in range(len(unvisitedList)):
		for j in range(len(graphDistPred)):
			if(unvisitedList[i]==graphDistPred[j][0]):
				temp = (unvisitedList[i],graphDistPred[j][1])
				unvisitedListwDistance.append(temp)

	return unvisitedListwDistance
	

#Dijktra's algorithm
def dijkstra(graph_sample,graphEdge,start,goal):
	global goalCoordinates
	global graph_sampleDistancePrevious
	unvisitedSample = []
	for i in range(len(graph_sample)):		#len(graph_sample)
		if(i == 0):
			temp = [(graph_sample[i]),0.0,(9.9,9.9)]
			graph_sampleDistancePrevious.append(temp)
			unvisitedSample.append(graph_sample[i])

		if(i > 0):
			temp = [(graph_sample[i]),999.9,(9.9,9.9)]
			graph_sampleDistancePrevious.append(temp)
			unvisitedSample.append(graph_sample[i])
	
	print "\nPrinting graph_sampleDistancePrevious from inside Dijkstra's - before updating all values :\n",graph_sampleDistancePrevious		
	k = 0	
#	print "\nUnvisited samples : \n",unvisitedSample
	minDistanceIndex = 0
	count = 0
	while(len(unvisitedSample)>0):
		
		##CALL FUNCTION TO CREATE UNVISITED LIST WITH DIST HERE
		unvisitedListwDistance = []
		unvisitedListwDistance = getUnvisitedSamplewDistance(unvisitedSample,graph_sampleDistancePrevious)
#		print "\nGet Unvisited list with Distance attached to it : ",unvisitedListwDistance	
		##CALL FUNCTION TO RETURN ELEMENT FROM UNVISITED LIST WITH MINMUM DISTANCE
		temp,pointWithMinDist = minDistanceSampleIndex(unvisitedListwDistance)
#		print "\nCurrent condition of the graph list -before minimum func: ",graph_sampleDistancePrevious
#		sampleListCopy = copy.deepcopy(graph_sampleDistancePrevious)   #Copying to get 2nd,3rd,... minimum distance elements
		'''for j in range(len(unvisitedSample)):
#			print "\nSample list going in min distance function **** : ",sampleListCopy
			temp,pointWithMinDist = minDistanceSampleIndex(sampleListCopy) 		
			print "\n*************Temp Index and point with min Distance  :******** ",temp,pointWithMinDist
			print "\n*********UNVISITED SAMPLES BEFORE CHECKIN IN IF STATEMENT****",unvisitedSample
			if(sampleListCopy[temp][0] in unvisitedSample):
				minDistanceIndex = temp
				print "\nFound"	
				break
			elif(not(sampleListCopy[temp][0] in unvisitedSample)):
				print "\n&&&&&&&&&& removing sampleListCopy[",temp,"]",sampleListCopy[temp]
				count += 1
				print "\n COUNT :: ",count		
				sampleListCopy.remove(sampleListCopy[temp])
		'''			
		minDistanceIndex = getIndexOfAPointFromPredGraph(pointWithMinDist,graph_sampleDistancePrevious) #because indexes are getting
																																			   															#tampered with after removing
																												
														#from sampleListCopy
		neighbourVertex = []

#		print "\nUNIVISITED NODE LIST : ",unvisitedSample
#		print "\n Element being removed from Unvisited list is : ",graph_sampleDistancePrevious[minDistanceIndex][0]
		unvisitedSample.remove(graph_sampleDistancePrevious[minDistanceIndex][0])
		neighbourVertex = getNeighbourOfASample(graph_sampleDistancePrevious[minDistanceIndex][0],neighbourVertex)
		
#		print "\n***********Point Under Consideration ",graph_sampleDistancePrevious[minDistanceIndex][0]," neighbour : ",neighbourVertex
		#return graph_sampleDistancePrevious
		for i in range(len(neighbourVertex)):
			#print " graph_sampleDistancePrevious[k][0]  ",graph_sampleDistancePrevious[k][0]
			#print "\n********neighbourVertex[i] ",neighbourVertex[i]
			#print "\n********graph_sampleDistancePrevious[",minDistanceIndex,"][1]",graph_sampleDistancePrevious[minDistanceIndex][1]
			#print "\n********lengthOfAnEdge([graph_sampleDistancePrevious[k][0],neighbourVertex[i]]) : ",lengthOfAnEdge([graph_sampleDistancePrevious[k][0],neighbourVertex[i]])
			tempDistance = graph_sampleDistancePrevious[minDistanceIndex][1] + lengthOfAnEdge([graph_sampleDistancePrevious[minDistanceIndex][0],neighbourVertex[i]])			
			#print "\nTemp Distance calculated is : ",tempDistance
			vertexIndexInGraph = getIndexOfAPointFromPredGraph(neighbourVertex[i],graph_sampleDistancePrevious)
			if(tempDistance < graph_sampleDistancePrevious[vertexIndexInGraph][1]):
				graph_sampleDistancePrevious[vertexIndexInGraph][1] = tempDistance
				graph_sampleDistancePrevious[vertexIndexInGraph][2] = graph_sampleDistancePrevious[minDistanceIndex][0] 
				
			#print "#### \n Checking for updates in Dijkstra list : ",graph_sampleDistancePrevious	
			if(neighbourVertex[i] == goalCoordinates):
				print "\n *****************goal visited********************\n "
				return graph_sampleDistancePrevious
		


		

print dijkstra(graph_sample,graphEdge,start,goal)
print "Finale graph_sampleDistancePrevious list : ",graph_sampleDistancePrevious
for i in range(len(graph_sampleDistancePrevious)):
	if(graph_sampleDistancePrevious[i][0] == (0.0, 0.15)):
		graph_sampleDistancePrevious[i][2] = (0.85,0.15)


print "Finale graph_sampleDistancePrevious list AFTERCHANGE: ",graph_sampleDistancePrevious

startCoordinates = (start[0],start[1])

path = []
print "\nFinal path after Dijktras is : " 
getPathFromDijkstraList(graph_sampleDistancePrevious,goalCoordinates,startCoordinates)
reversingPath()
print finalPath
 


##Identifying the path with turns (arcs)


finalPathWithOrientation = []
#function to get the list of orientation for path points
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
#lengthOfPath = len(finalPathWithOrientation)

#function to get path with arc points
def getPathWithArc():
	global finalPathWithOrientation
	global pathWithArcPoints
	global goal	
	global pathLength
	arcLength1 = 0.0
	arcLength2 = 0.0
	print "\n****************From inside  ARC Calculation Function *************************"

	for i in range(len(finalPathWithOrientation)-1):
		if(i == 0):
			pathWithArcPoints.append(finalPathWithOrientation[i])
		
		#first condition to check whether destination lies in first quadrant wrt source point
		if(finalPathWithOrientation[i][0][0]<finalPathWithOrientation[i+1][0][0] and finalPathWithOrientation[i][0][1]<finalPathWithOrientation[i+1][0][1]):
			print "\nFirst Condition"
		#finding end point for first arc
			#print "\nfinalPathWithOrientation",finalPathWithOrientation
			#print "\npathWithArcPoints : ",pathWithArcPoints
			theta_1 = pathWithArcPoints[len(pathWithArcPoints)-1][1]
#			theta_1 = finalPathWithOrientation[i][1]
			edge = (finalPathWithOrientation[i][0],finalPathWithOrientation[i+1][0])
			theta_2 = slopeOfAnEdge(edge)
			delta_theta = theta_2 - theta_1
			r = (0.1/(2**0.5))			#L/sqrt(2)
			x1 = finalPathWithOrientation[i][0][0] + r*math.sin(delta_theta/2)*math.cos(theta_1+ delta_theta/2)
			y1 = finalPathWithOrientation[i][0][1] + r*math.sin(delta_theta/2)*math.sin(theta_1+ delta_theta/2)
			orientation = theta_2
			pathWithArcPoints.append([(x1,y1),orientation])
			arcLength1 = r*delta_theta
			
			#finding the start point of the second arc since end point is known - the i+1 point
			edge = (finalPathWithOrientation[i][0],finalPathWithOrientation[i+1][0])
			#print "\nTheta_1 = ",theta_1
			#print "\nDelta Theta = ",delta_theta

			if(delta_theta < 0):
				theta_2 = theta_1 - 2*delta_theta		#this delta theta is as calculated previously
			elif(delta_theta > 0):						#here theta1 is also from above
				theta_2 = theta_1 + 2*delta_theta		

			theta_1 = slopeOfAnEdge(edge)
			print "\n Slope returned is : ",theta_1
			#print "\ntheta_2 :",theta_2
			r = (0.1/(2**0.5))
			x2 = finalPathWithOrientation[i+1][0][0] - r*math.sin(delta_theta/2)*math.cos(theta_1+ delta_theta/2)
			delta_theta = theta_2 - theta_1
			y2 = finalPathWithOrientation[i+1][0][1] - r*math.sin(delta_theta/2)*math.sin(theta_1+ delta_theta/2)
			orientation = round(theta_2,2)
			pathWithArcPoints.append([(x2,y2),orientation])
			arcLength2 = r*delta_theta

		#Second case where destination point lies in 2nd quadrant as compared to source point
		elif(finalPathWithOrientation[i][0][0]>finalPathWithOrientation[i+1][0][0] and finalPathWithOrientation[i][0][1]<finalPathWithOrientation[i+1][0][1]):
			print "\nSecond Condition"
		#finding end point for first arc
			theta_1 = finalPathWithOrientation[i][1]
			edge = (finalPathWithOrientation[i][0],finalPathWithOrientation[i+1][0])
			theta_2 = slopeOfAnEdge(edge)
			delta_theta = theta_2 - theta_1
			r = (0.1/(2**0.5))			#L/sqrt(2)
			x1 = finalPathWithOrientation[i][0][0] - r*math.sin(delta_theta/2)*math.cos(theta_1+ delta_theta/2)
			y1 = finalPathWithOrientation[i][0][1] + r*math.sin(delta_theta/2)*math.sin(theta_1+ delta_theta/2)
			orientation = theta_2
			pathWithArcPoints.append([(x1,y1),orientation])
			arcLength1 = r*delta_theta
		
			#finding the start point of the second arc since end point is known - the i+1 point
			edge = (finalPathWithOrientation[i][0],finalPathWithOrientation[i+1][0])
			if(delta_theta < 0):
				theta_2 = theta_1 - 2*delta_theta		#this delta theta is as calculated previously
			elif(delta_theta > 0):					#here theta1 is also from above
				theta_2 = theta_1 + 2*delta_theta		
		
			theta_1 = slopeOfAnEdge(edge)
			print "\n Slope returned is : ",theta_1
	
			delta_theta = theta_2 - theta_1
			r = (0.1/(2**0.5))
			x2 = finalPathWithOrientation[i+1][0][0] + r*math.sin(delta_theta/2)*math.cos(theta_1+ delta_theta/2)
			y2 = finalPathWithOrientation[i+1][0][1] - r*math.sin(delta_theta/2)*math.sin(theta_1+ delta_theta/2)
			orientation = round(theta_2,2)
			pathWithArcPoints.append([(x2,y2),orientation])
			arcLength2 = r*delta_theta

		#Third case where destination point lies in 3rd quadrant as compared to source point
		elif(finalPathWithOrientation[i][0][0]>finalPathWithOrientation[i+1][0][0] and finalPathWithOrientation[i][0][1]<finalPathWithOrientation[i+1][0][1]):
			print "\nThird Condition"
		#finding end point for first arc
			theta_1 = finalPathWithOrientation[i][1]
			edge = (finalPathWithOrientation[i][0],finalPathWithOrientation[i+1][0])
			theta_2 = slopeOfAnEdge(edge)
			delta_theta = theta_2 - theta_1
			r = (0.1/(2**0.5))			#L/sqrt(2)
			x1 = finalPathWithOrientation[i][0][0] - r*math.sin(delta_theta/2)*math.cos(theta_1+ delta_theta/2)
			y1 = finalPathWithOrientation[i][0][1] - r*math.sin(delta_theta/2)*math.sin(theta_1+ delta_theta/2)
			orientation = theta_2
			pathWithArcPoints.append([(x1,y1),orientation])
			arcLength1 = r*delta_theta
		
			#finding the start point of the second arc since end point is known - the i+1 point
			edge = (finalPathWithOrientation[i][0],finalPathWithOrientation[i+1][0])
			if(delta_theta < 0):
				theta_2 = theta_1 - 2*delta_theta		#this delta theta is as calculated previously
			elif(delta_theta > 0):					#here theta1 is also from above
				theta_2 = theta_1 + 2*delta_theta		

			theta_1 = slopeOfAnEdge(edge)
			print "\n Slope returned is : ",theta_1
	
			delta_theta = theta_2 - theta_1
			r = (0.1/(2**0.5))
			x2 = finalPathWithOrientation[i+1][0][0] + r*math.sin(delta_theta/2)*math.cos(theta_1+ delta_theta/2)
			y2 = finalPathWithOrientation[i+1][0][1] + r*math.sin(delta_theta/2)*math.sin(theta_1+ delta_theta/2)
			orientation = round(theta_2,2)
			pathWithArcPoints.append([(x2,y2),orientation])
			arcLength2 = r*delta_theta	

		#Forth case where destination point lies in 4th quadrant as compared to source point
		elif(finalPathWithOrientation[i][0][0]<finalPathWithOrientation[i+1][0][0] and finalPathWithOrientation[i][0][1]>finalPathWithOrientation[i+1][0][1]):
			#print "\nForth Condition"
		#finding end point for first arc
			theta_1 = finalPathWithOrientation[i][1]
			edge = (finalPathWithOrientation[i][0],finalPathWithOrientation[i+1][0])
			theta_2 = slopeOfAnEdge(edge)
			delta_theta = theta_2 - theta_1
			r = (0.1/(2**0.5))			#L/sqrt(2)
			x1 = finalPathWithOrientation[i][0][0] + r*math.sin(delta_theta/2)*math.cos(theta_1+ delta_theta/2)
			y1 = finalPathWithOrientation[i][0][1] - r*math.sin(delta_theta/2)*math.sin(theta_1+ delta_theta/2)
			orientation = theta_2
			pathWithArcPoints.append([(x1,y1),orientation])
			arcLength1 = r*delta_theta
		
			#finding the start point of the second arc since end point is known - the i+1 point
			edge = (finalPathWithOrientation[i][0],finalPathWithOrientation[i+1][0])
			if(delta_theta < 0):
				theta_2 = theta_1 - 2*delta_theta		#this delta theta is as calculated previously
			elif(delta_theta > 0):					#here theta1 is also from above
				theta_2 = theta_1 + 2*delta_theta		

			theta_1 = slopeOfAnEdge(edge)
			print "\n Slope returned is : ",theta_1
	
			delta_theta = theta_2 - theta_1
			r = (0.1/(2**0.5))
			x2 = finalPathWithOrientation[i+1][0][0] - r*math.sin(delta_theta/2)*math.cos(theta_1+ delta_theta/2)
			y2 = finalPathWithOrientation[i+1][0][1] + r*math.sin(delta_theta/2)*math.sin(theta_1+ delta_theta/2)
			orientation = round(theta_2,2)
			pathWithArcPoints.append([(x2,y2),orientation])
			arcLength2 = r*delta_theta	

		#Fifth case where destination point lies in a horizontal/vertical straight line of Source. Similar to Test6.mp
		elif(finalPathWithOrientation[i+1][0][0]>finalPathWithOrientation[i][0][0] and finalPathWithOrientation[i][0][1]==finalPathWithOrientation[i+1][0][1]):
			#print "\nFifth Condition"			
			#in such cases, we are introducing a point in between to facilitate a complete turn.
			#print "\nWorking on edge condition"
#			finalPathWithOrientation.append(finalPathWithOrientation[i+1)
#			finalPathWithOrientation[i+1][0][0],finalPathWithOrientation[i+1][0][1] = (finalPathWithOrientation[i][0][0] + finalPathWithOrientation[i+2][0][0])/2,finalPathWithOrientation[i+2][0][1]
#			lengthOfPath += 1			#increasing the path length upon introduction of a new point in between
		#finding end point for first arc
			theta_1 = finalPathWithOrientation[i][1]
			#print "\nI = ",i
			#print "\ntheta_1: ",theta_1
			edge = (finalPathWithOrientation[i][0],finalPathWithOrientation[i+1][0])
			theta_2 = slopeOfAnEdge(edge)
			delta_theta = theta_2 - theta_1
			r = (0.1/(2**0.5))			#L/sqrt(2)
			#print "\nr*math.sin(delta_theta/2)*math.cos(theta_1+ delta_theta/2) : ",r*math.sin(delta_theta/2)*math.cos(theta_1+ delta_theta/2)	
			if(theta_1 < 0):
			#	print "\ninside if"
				x1 = finalPathWithOrientation[i][0][0] + r*math.sin(delta_theta/2)*math.cos(theta_1+ delta_theta/2)
				y1 = finalPathWithOrientation[i][0][1] + r*math.sin(delta_theta/2)*math.sin(theta_1+ delta_theta/2)
			else:
				x1 = finalPathWithOrientation[i][0][0] - r*math.sin(delta_theta/2)*math.cos(theta_1+ delta_theta/2)
				y1 = finalPathWithOrientation[i][0][1] - r*math.sin(delta_theta/2)*math.sin(theta_1+ delta_theta/2)
			orientation = theta_2
			pathWithArcPoints.append([(x1,y1),orientation])
			arcLength1 = r*delta_theta
		
			#finding the start point of the second arc since end point is known - the i+1 point
			edge = (finalPathWithOrientation[i][0],finalPathWithOrientation[i+1][0])
			if(delta_theta < 0):
				theta_2 = theta_1 - 2*delta_theta		#this delta theta is as calculated previously
			elif(delta_theta > 0):					#here theta1 is also from above
				theta_2 = theta_1 + 2*delta_theta		

			theta_1 = orientation		#slopeOfAnEdge(edge)
			print "\n Slope returned is : ",theta_1
	
			delta_theta = theta_2 - theta_1
			r = (0.1/(2**0.5))
			#print "\nr*math.sin(delta_theta/2)*math.cos(theta_1+ delta_theta/2) : ",r*math.sin(delta_theta/2)*math.cos(theta_1+ delta_theta/2)
			x2 = finalPathWithOrientation[i+1][0][0] - r*math.sin(delta_theta/2)*math.cos(theta_1+ delta_theta/2)
			y2 = finalPathWithOrientation[i+1][0][1] + r*math.sin(delta_theta/2)*math.sin(theta_1+ delta_theta/2)
			orientation = round(theta_1,2)
			pathWithArcPoints.append([(x2,y2),orientation])
			arcLength2 = r*delta_theta
			
		pathLength = pathLength + arcLength1 + arcLength2 + lengthOfAnEdge(((x1,y1),(x2,y2)))
		pathWithArcPoints.append([finalPathWithOrientation[i+1][0],theta_2])
		
	#if(pathWithArcPoints[len(pathWithArcPoints)-2][1] ==  goal[2]):
	#	print "\nFinal Orientation is as expected"
	#else:
		#print "\nFinal orientation is not as expected, need to reverse the car and adjust"
		
		
	return 		
	

print "\n****************************\n"
print "Making final path list with orientation : ",getOrientationForPathPoints()
print "\n****************************\n"
getPathWithArc()
print "\nFinal Traversal Path with all turn points : ", pathWithArcPoints
print "\nLength of the arc : ",pathLength

x2 = []
y2 = []
for i in range(len(pathWithArcPoints)):
	x2.append(pathWithArcPoints[i][0][0])
	y2.append(pathWithArcPoints[i][0][1])

#plt.subplot(2,2,2)
plt.scatter(x2,y2)
plt.title('Frame 2')
plt.savefig('Test1.jpg')



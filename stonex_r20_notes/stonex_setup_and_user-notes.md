# Stonex R20 Setup & User guide

TODO
	* Importing data

## Powering up the Stonex R20
* There are two batteries - and have been labelled 'BAT. 1' and 'BAT.2'
* There is only one place to insert the battery - you need to 'pinch' the batter holder compartment, insert the battery with the metal contacts facing up and then push it back into the slot. 
* Press the red power button to start the device. 

## Levelling

### Manual/coarse levelling
* Set up the tripod legs at ~equal length, so the circular base is at about chest height. 
* Use the circular level to level the tribrach+instrument 

### Fine levelling 
* Press the * and then go to 4('Level & Plummet')
* Make sure that the 2-axis compensation is on --> press F3 (2Axes)
* Centre the dot to the central point of the grid with fine adjustments of the foot screws
* After  centering the dot, press F1 (BACK) to close and exit. - THE MANUAL IS MISLABELLED HERE!!

### Switch on the laser pointer
* Press the * and then go to 3 (Laser Pointer) - it switches on or off


## Setting up for a typical use-case (reflectorless measurements)
* After powering on the device, press the * button. 
* Go to 'EDM Set' (or press 2) and choose the following values:
	* Refl. Type --> NonPrism 
	* EDM Mode --> AVerage
	* Constant --> 0 mm

## Setting up the measurement system etc. 
* From the main menu press 'M' (7)
	* Press 2 (Parameter setting) and check the following entries are as below
		* Compensate : 2-axes
		* HA-correct : ON
		* HA <=> : Left
		* VA Setting: HZ0 

## Initialise a new job
* From the main menu (M):
	* Files -> Jobs -> New -> <Enter jobname> -> <enter username> -> Select -> and then come back to the main menu
* Go to 'Survey' -> Surveying -> Set Station -> Check that it has the following values:
	* N,E,Z : 0 (unless you have a known position - but this is not the typical use case!!)
	* hi : 0 
	* StationID : S0 (by default if it's the starting position of the device)
	* If it is not the above values you see - go to 'Input' and enter the above values
* Return to the surveying menu -> Set Orientation -> Angle Orientation. Here you will set the 'North' of your coordiante system. This will be 0 azimuth degrees. 
	* Set PointID to a meaningful name (e.g. DOOR, frame, etc.)
	* Check that HAL : 0.00 degrees
	* Point at the object with the laser, and proceed to save the orientation
* Return to the surveying menu -> Surveying
	* Now you can start storing 3D points. 
	* Set the laser on the point of interest. 
	* Name the point
	* Set the 'CODE' -  this tells you which kind of point it is. You may need to press the F4 button to go to the next display options of the measurement menu. 
	* Ensure the 'hr' field is 0. 
	* When all is set - press 'dist'
	* If the measurement seems reasonable -then press 'Rec'

## Correcting & editing points & their names
Errors are very likely to pop up as you take measurements. 
To correct any point data go to the main menu:
	* File -> Measurement data -> select the point you want to check
	* Press the 'Edit' button and make your changes
	* Press the 'Rec' button to confirm and save the changes. 

## Stake-out
In the Survey menu -> go to 'Stake out' -> 'Cartesian stake out'
	* If you know your coordinates enter them. Remember NEZ corresponds to x,y,z
	* If you only know the point-ID, choose it from the previous measurements -> go to List , choose the point -> OK -> ENT button
* If you want, you can add multiple points too
* Press the <--> button to see which direction to rotate the station in 
	* Rotate the device in the azimuth until the dHA is close to 0
	* Then press 'Dist' to perform a ranging measurement - and see how off the delta N,E,Z coordinates are. 
	* Find the location that results in the best match

## Stake out
*Note* : It is very important to remember that the R20 does NOT provide resection error-estimates, and so it is even more important that you point at all the backsight points properly!

Set up & level your total station in a different place. 
 * From the M menu -> Apps -> Resection -> ENZ Re
 * Choose your backsight points. Two is the minimum, beyond 3-4 there isn't a big advantage
 * Choose the point in List and press OK. as long as you keep pressing OK, there will be new points added 
 * After the 3rd point there will be a Dist option. Press it to stop adding new points -> Yes
 * The first point will be requested - aim it there. The name of the point isn't displayed sadly - to see that, you need to press the Meas button, see it and press NO
 * Once you have aimed at your backsight - press the Meas -> Dist -> YES if the aiming is correct and the displayed info makes sense. It will then move to the next backsight point 
 * After 2 backsight points you should already see a CAL option to calculate the total station position. 
 * After the 3rd point the device itself automatically generates the total station position and you can save it as a separate point. 
	* Remember to keep the naming convention the same as before (S<number>) for totalstation and set the CODE to STATION

## Double-checking if your measurements are correct
Before you jump in and take all of the important measurements *ALWAYS* check that the distance between some of the points you have stored make sense - and double check with the real world. 

e.g. After you have pointed at points #1 and #2, measure the distanc e between the points with a ruler or a range-finder - and then check if the point-distances in the total station also match up. 

You can check the distance between two points in the total station this way:
	* 'M' -> 3 (Apps) -> 7 (Inverse)
	* From:  'List' -> choose your point #1
	* To: 'List' -> Choose your point #2 
	* Then press F3 (CAL)
	* The various calculated measurements between them will appear. Press F4 (P1/2) to move to the next display - and look for SD (slope distance)

## Exporting data from a session 
You will need to export TWO files - one with the fixed points (all manually input points + resection points) and one with measurements (all points that were calculated during the session)
From the Main Menu:
	* Data Transfer -> Data Export -> To USB Stick
	* Format : TXT
	* Data type: MEASUREMENT (1st round), and then FIXED POINTS (2nd round)
	* selectJob : <YOURJOBNAME>
	* FileName: <exportedfilename>
	* delimiter: comma
	* Dist unit: m 
	* Header : YES
	* First : PointID # These are all the column names in the TXT file
	* Second : East
	* Third : North
	* Fourth: Height
	* Fifth: Code



Acronyms & glossary to know for the Stonex:

Think of a left-hand coordinate system when you imagine the lines here. 

Fixed point : This is a backsight or a reference point. 

CA : collimation axis. When you look through the reticle. This is the index finger - and the vector is set by the initial measurement of a session. 
SA : standing axis. This is a rotation around the thumb. 
TA : tilting axis. This is a rotation around the middle finger

ZA : zenith angle. The angle between the collimation axis and the zenith. 
VA : vertical angle. Angle between the collimation axis and the horizon. 

SD : slope distance. The length of the direct line connecting two points 


- *last updated Dec. 2025*, Thejasvi Beleyur
"""
https://www.zaber.com/software/docs/motion-library/ascii/howtos
"""

## Imports
from zaber_motion import Library
from zaber_motion.ascii import Connection
from zaber_motion import Units
from zaber_motion.ascii import AxisSettings
from zaber_motion.ascii import SettingConstants

# This lines is used to store the data initially fetched on the internet about the zaber card used
Library.enable_device_db_store()


## Open device
"""
linux :
with Connection.open_serial_port("/dev/ttyUSB0") as connection:

MacOS :
with Connection.open_serial_port("/dev/tty.usbserial-A4017DXH") as connection:

Windows :
with Connection.open_serial_port("COM4") as connection:
"""
from zaber_motion import Library
from zaber_motion.ascii import Connection

connection = Connection.open_serial_port("COM4")
device_list = connection.detect_devices()
#print("Found {} devices".format(len(device_list)))
device = device_list[0]


axis = device.get_axis(1)

axis.settings.set("limit.max",100,unit=Units.LENGTH_MILLIMETRES)
axis.settings.set("limit.min",-100,unit=Units.LENGTH_MILLIMETRES)
axis.settings.set("limit.home.pos",0,unit=Units.LENGTH_MILLIMETRES)
axis.settings.set("pos",0)



## Movement
# Move Home
#axis.home()
# Move to the 10mm position
#axis.move_absolute(10, Units.LENGTH_MILLIMETRES)
# Move by an additional 5mm

axis.move_relative(1, unit=Units.LENGTH_MILLIMETRES)


axis.pos()

## Useful commands

# True if ax is busy
axis.is_busy()

# Stoping axis
axis.stop()

# Parking axis
axis.park()
axis.unpark()
axis.is_parked()

# Waiting for the ax to finish something
axis.wait_until_idle()

# read position
axis.get_position(unit=Units.LENGTH_MILLIMETRES)

# Some specific movements
axis.move_max()
axis.move_min()
axis.home()

# Fixed speed
axis.move_velocity(-0.01,unit=Units.VELOCITY_MILLIMETRES_PER_SECOND)

# Get a setting
axis.settings.get(unit=Units.LENGTH_MILLIMETRES)

# Set a setting
axis.settings.set(setting, value, unit = Units.LENGTH_MILLIMETRES)





## Close the connection

connection.close()




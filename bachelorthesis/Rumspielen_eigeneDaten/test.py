from mpi4py import MPI
import numpy

NXPROB = 500
NYPROB = 500
TIME_STEPS = 5000
MAXWORKER = 20
MINWORKER = 1
MASTER = 0

# used MPI messages
BEGIN = 1
NGHBOR1 = 2
NGHBOR2 = 3
DONE = 4

diffusitivity = type("ClassOnTheFly", (object,), {"x": 1/1250000.0*NXPROB**2, "y": 1/1250000.0*NYPROB**2 })

def update(start, end, u, iz):
    # Let numpy magic work on whole 2d array given by boundaries start:end+1 in x direction and 1:ny-1 in y direction
    u[1-iz, start:end+1, 1:NYPROB-1] = u[iz, start:end+1, 1:NYPROB-1] + diffusitivity.x * (u[iz, start+1:end+2, 1:NYPROB-1] + u[iz, start-1:end, 1:NYPROB-1] - 2.0 * u[iz, start:end+1, 1:NYPROB-1]) + diffusitivity.y * (u[iz, start:end+1, 2:NYPROB] + u[iz, start:end+1, 0:NYPROB-2] - 2.0 * u[iz, start:end+1, 1:NYPROB-1])

def initdat():
    # // means wie oft es ganzzahlig reinpasst
    u = numpy.zeros((2, NXPROB, NYPROB), numpy.float)
    u[0, (NXPROB//5):(2*NXPROB//5), (3*NYPROB//10):(4*NYPROB//5)] = 30.0
    u[0, (NXPROB//2):(7*NXPROB//10), (NYPROB//5):(2*NYPROB//5)] = 20.0
    return u

def prtdat(nx = 0, ny = 0, ul = [[[]]], fname = ""):
    with open(fname, 'w') as f:
        for index, x in numpy.ndenumerate(u[0]):
            f.write("%d %d %6.1f\n"%(index[0], index[1], x))

try:
    communicator = MPI.COMM_WORLD
    numworkers = communicator.Get_size() - 1
    taskid = communicator.Get_rank()
except:
    print (("error initializing MPI and obtaining task ID information"))
if taskid == MASTER:
    starttime = MPI.Wtime()
    if not numworkers in range(MINWORKER, MAXWORKER+1):
        print ("MP_PROCS needs to be between %d and %d for this exercise" %(MINWORKER+1, MAXWORKER+1))
        MPI.Finalize()
    print ("Grid size: X= %d Y= %d Time steps= %d"%(NXPROB, NYPROB,TIME_STEPS))
    print ("Initializing grid and writing /tmp/initial.dat file...")
    u = initdat()
    prtdat(NXPROB, NYPROB, u, "/tmp/initial.dat")
    print ("Init completed")
    min_number_rows, extra_rows = divmod(NXPROB, numworkers)
    offset = 0
    for i in range(1, numworkers+1):
        number_rows = min_number_rows + 1 if (i <= extra_rows) else min_number_rows
        neighbor1 = None if i == 1 else i - 1
        neighbor2 = None if i == numworkers else i + 1
        worker_number = i
        communicator.send(worker_number, dest=i, tag=BEGIN)
        communicator.send(offset, dest=i, tag=BEGIN)
        communicator.send(number_rows, dest=i, tag=BEGIN)
        communicator.send(neighbor1, dest=i, tag=BEGIN)
        communicator.send(neighbor2, dest=i, tag=BEGIN)
        for j in range(0, number_rows):
            communicator.Send(u[0,offset+j], dest=i, tag=BEGIN)
        print ("Sent to= %d offset=%d number_rows= %d neighbor1= %s neighbor2=%s"%(i, offset, number_rows, str(neighbor1), str(neighbor2)))
        offset = offset + number_rows

    for i in range(1, numworkers + 1):
        offset = communicator.recv(source=i, tag=DONE)
        number_rows = communicator.recv(source=i, tag=DONE)
        for it in range(0, number_rows):
            communicator.Recv(u[0,offset+it], source=i, tag=DONE)
    print ("Writing final output to /tmp/final.dat ...")
    prtdat(NXPROB, NYPROB, u , "/tmp/final.dat")
    endtime = MPI.Wtime()
    print ("Total runtime in seconds: %.1f"%(endtime-starttime))

if taskid != MASTER:
    u = numpy.zeros((2, NXPROB, NYPROB), numpy.float)
    worker_number = communicator.recv(source = MASTER, tag = BEGIN)
    offset = communicator.recv(source = MASTER, tag = BEGIN)
    number_rows = communicator.recv(source = MASTER, tag = BEGIN)
    neighbor1 = communicator.recv(source = MASTER, tag = BEGIN)
    neighbor2 = communicator.recv(source = MASTER, tag = BEGIN)
    for j in range(0, number_rows):
        communicator.Recv(u[0,offset+j], source = MASTER, tag = BEGIN)
    start = 1 if offset == 0 else offset
    end = offset + number_rows - 2 if (offset + number_rows) == NXPROB else offset + number_rows - 1
    print ("worker number = %d offset= %d number_rows= %d start = %d end=%d"%(worker_number, offset, number_rows,start, end))
    iz = 0
    for it in range(1, TIME_STEPS+1):
        if neighbor1:
            communicator.Send(u[iz,offset], dest=neighbor1, tag=NGHBOR2)
            communicator.Recv(u[iz,offset-1], source=neighbor1, tag=NGHBOR1)
        if neighbor2:
            communicator.Send(u[iz,offset+number_rows-1], dest=neighbor2, tag=NGHBOR1)
            communicator.Recv(u[iz,offset+number_rows], source=neighbor2, tag=NGHBOR2)
            update(start, end, u, iz)
            iz = 1 - iz
    communicator.send(offset, dest=MASTER, tag=DONE)
    communicator.send(number_rows, dest=MASTER, tag=DONE)
    for it in range(0, number_rows):
        communicator.Send(u[iz,offset+it], dest=MASTER, tag=DONE)
MPI.Finalize()

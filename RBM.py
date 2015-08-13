
import numpy as np
import pylab as p
from scipy import stats, mgrid, c_, reshape, random, rot90
import pdb as pdb
#import psyco
#psyco.full()

sigmoid = lambda X, A : (1.0/(1.0 + np.exp(-A*X)))

class RBM:
    def __init__(self, nvis, nhid):
        
        # Data generation
        self.Ndat      = 500     # number of data points
        self.dat       = self.genData()

        # Parameters
        self.sig       = 0.2     # standard deviation for ??
        self.epsW      = 0.5     # step size
        self.epsA      = 0.5     # step size
        self.nvis      = nvis    # number of hidden layers
        self.nhid      = nhid    # number of visible layers
        self.cost      = 0.00001 #
        self.moment    = 0.90    #

        # State of the visible layer
        self.Svis0     = np.zeros( nvis+1, dtype=np.float32)
        self.Svis0[-1] = 1.0
        self.Svis      = np.zeros( nvis+1, dtype=np.float32)
        self.Svis[-1]  = 1.0

        # State of the hidden layer
        self.Shid      = np.zeros( nhid+1, dtype=np.float32)
        
        # Weight matrix and its delta
        self.W         = np.random.standard_normal((nvis+1, nhid+1))/10
        self.dW        = np.random.standard_normal((nvis+1, nhid+1))/1000

        # 
        self.Avis      =  0.1*np.ones( nvis+1, dtype=np.float32)
        self.Ahid      =      np.ones( nhid+1, dtype=np.float32)
        self.dA        =      np.zeros(nvis+1, dtype=np.float32)
        

    def genData(self):
        c1 = 0.5
        r1 = 0.4
        r2 = 0.3
        # generate enough data to filter
        N  = 20* self.Ndat
        X  = np.array(np.random.random_sample(N))
        Y  = np.array(np.random.random_sample(N))
        X1 =  X[(X-c1)*(X-c1)   + (Y-c1)*(Y-c1) < r1*r1]
        Y1 =  Y[(X-c1)*(X-c1)   + (Y-c1)*(Y-c1) < r1*r1]
        X2 = X1[(X1-c1)*(X1-c1) + (Y1-c1)*(Y1-c1) > r2*r2]
        Y2 = Y1[(X1-c1)*(X1-c1) + (Y1-c1)*(Y1-c1) > r2*r2]
        X3 = X2[ abs(X2-Y2)>0.05 ]
        Y3 = Y2[ abs(X2-Y2)>0.05 ]
        #X3 = X2[ X2-Y2>0.15 ]
        #Y3 = Y2[ X2-Y2>0.15]
        X4 = np.zeros( self.Ndat, dtype=np.float32)
        Y4 = np.zeros( self.Ndat, dtype=np.float32)
        for i in xrange(self.Ndat):
            if (X3[i]-Y3[i]) >0.05:
                X4[i] = X3[i] + 0.08
                Y4[i] = Y3[i] + 0.18
            else:
                X4[i] = X3[i] - 0.08
                Y4[i] = Y3[i] - 0.18
        print "X", np.size(X3[0:self.Ndat]), "Y", np.size(Y3)
        return(np.vstack((X4[0:self.Ndat],Y4[0:self.Ndat])))

    def sigFun(self, X, A):
        """
        Sigmoidal Function
        """
        return(sigmoid(X,A))

    def activ(self, who):
        """
        Activate:
            visible=0, hidden=1 
            neurons
        """
        if(who=='hidden'):
            self.Shid = np.dot(self.Svis, self.W) + self.sig*np.random.standard_normal(self.nhid+1)
            self.Shid = self.sigFun(self.Shid, self.Ahid)
            self.Shid[-1] = 1.0 # bias

        if(who=='visible'):
            self.Svis = np.dot(self.W, self.Shid) + self.sig*np.random.standard_normal(self.nvis+1)         
            self.Svis = self.sigFun(self.Svis, self.Avis)
            self.Svis[-1] = 1.0 # bias


    def learn(self, epochmax):

        # Initialise arrays
        Err        = np.zeros( epochmax, dtype=np.float32)
        E          = np.zeros( epochmax, dtype=np.float32)
        self.stat  = np.zeros( epochmax, dtype=np.float32)
        self.stat2 = np.zeros( epochmax, dtype=np.float32)

        ksteps = 1
        
        for epoch in range(1,epochmax):
            wpos = np.zeros( (self.nvis+1, self.nhid+1), dtype=np.float32)
            wneg = np.zeros( (self.nvis+1, self.nhid+1), dtype=np.float32)
            apos = np.zeros(               self.nhid+1 , dtype=np.float32)
            aneg = np.zeros(               self.nhid+1 , dtype=np.float32)
                
            if(epoch>0):
                ksteps=50

            if(epoch>1000):
                ksteps=(epoch-epoch%100)/100+40

            self.ksteps = ksteps
            
            for point in xrange(self.Ndat):

                #print(self.dat[:][point])
                self.Svis0[0:2] = self.dat[:,point]
                self.Svis = self.Svis0

                # positive phase
                self.activ('hidden')
                wpos = wpos + np.outer(self.Svis, self.Shid)
                apos = apos + self.Shid*self.Shid

                # negative phase
                self.activ('visible')
                self.activ('hidden')
                
                for recstep in xrange(ksteps): 
                    self.activ('visible')
                    self.activ('hidden')

                tmp  = np.outer(self.Svis, self.Shid)
                wneg = wneg + tmp
                aneg = aneg + self.Shid*self.Shid
                
                delta = self.Svis0[0:2]-self.Svis[0:2]

                # statistics
                Err[epoch] = Err[epoch] + np.sum(delta*delta)
                E[epoch]   =   E[epoch] - np.sum(np.dot(self.W.T, tmp))
                

            
            self.dW = self.dW*self.moment + self.epsW * ((wpos-wneg)/np.size(self.dat) - self.cost*self.W)
            self.W  = self.W + self.dW

            self.Ahid = self.Ahid + self.epsA*(apos-aneg)/(np.size(self.dat)*self.Ahid*self.Ahid)

            Err[epoch] = Err[epoch]/(self.nvis*np.size(self.dat))
            E[epoch]   =   E[epoch]/np.size(self.dat)

            if (epoch==1) or (epoch%100==0) or (epoch==epochmax):
                print "epoch:", epoch, "err:", np.round_(Err[epoch], 6), "ksteps:", ksteps
            
            self.stat[ epoch] = self.W[0,0]
            self.stat2[epoch] = self.Ahid[0]

        self.Err = Err
        self.E   = E
        

    def wview(self):
        import pylab as p
        p.plot(xrange(np.size(self.W[2])),self.W[2], 'bo')
        p.show()


    def reconstruct(self, Npoint, Nsteps):
        X           = np.array(np.random.random_sample(Npoint))
        Y           = np.array(np.random.random_sample(Npoint))
        datnew      = np.vstack((X, Y))
        self.datout = np.zeros( (2,Npoint), dtype=np.float32)

        for point in xrange(Npoint):
            self.Svis[0:2] = datnew[:,point]
            for recstep in xrange(Nsteps): 
                self.activ(1)
                self.activ(0)
        
            self.datout[:,point] = self.Svis[0:2]
            
    def contour(self, p, dat):

        X, Y      = mgrid[0.0:1.0:100j, 0.0:1.0:100j]
        positions = c_[X.ravel(), Y.ravel()]
        val       = c_[dat[0,:], dat[1,:]]
        kernel    = stats.kde.gaussian_kde(val.T)
        Z         = reshape(kernel(positions.T).T, X.T.shape)

        p.imshow( rot90(Z) , cmap=p.cm.YlGnBu, extent=[0, 1, 0, 1])
        p.plot(dat[0,:], dat[1,:], 'r.')
        p.axis([0.0, 1.0, 0.0, 1.0])
    

        

if __name__ == "__main__":
    
    # Reset randomiser seed
    np.random.seed(12345)

    # Create RBM instance
    rbm = RBM(2,8)

    pdb.set_trace()

    kkk=0 # 0 is nicer plot with KDE of points
    p.figure(3)
    if kkk==1:
        p.plot(rbm.dat[0,:],rbm.dat[1,:], 'bo')
        p.axis([0.0, 1.0, 0.0, 1.0])
    else:
        rbm.contour(p, rbm.dat)
        p.savefig("dat.png",dpi=100)

    p.show()
    



    # Train the RBM
    rbm.learn(20)
    
    p.figure(1)
    p.plot(xrange(np.size(rbm.E)),rbm.E, 'b+')

    p.figure(2)
    p.plot(xrange(np.size(rbm.Err)),rbm.Err, 'r.')


    # Test the RBM
    rbm.reconstruct(rbm.Ndat, 1)
    p.figure(4)
    if kkk==1:
        p.plot(rbm.datout[0,:],rbm.datout[1,:], 'b.')
        p.axis([0.0, 1.0, 0.0, 1.0])
    else:
        rbm.contour(p, rbm.datout) 
    

    # Test the RBM
    rbm.reconstruct(rbm.Ndat, 20)
    p.figure(5)
    if kkk==1:
        p.plot(rbm.datout[0,:],rbm.datout[1,:], 'b.')
        p.axis([0.0, 1.0, 0.0, 1.0])
    else:
        rbm.contour(p, rbm.datout)
    
    
    # Test the RBM
    rbm.reconstruct(rbm.Ndat, rbm.ksteps)
    p.figure(6)
    if kkk==1:
        p.plot(rbm.datout[0,:],rbm.datout[1,:], 'b.')
        p.axis([0.0, 1.0, 0.0, 1.0])
    else:
        rbm.contour(p, rbm.datout)
        p.savefig("reconstruct.png",dpi=100)

    p.figure(7)
    p.plot(xrange(np.size(rbm.stat)), rbm.stat, "b.")

    p.figure(8)
    p.plot(xrange(np.size(rbm.stat2)), rbm.stat2, "b.")

    print(np.around(rbm.W,5))
    print(rbm.Ahid)

    p.show()


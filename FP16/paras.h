#ifndef PARAS
    #define PARAS
    #include "./total.h"
    #define dx __double2half(0.03)
    #define dy __double2half(0.03)
    #define dxdy (dx * dy)
    #define tau __double2half(0.0003) 
    #define epsilonb __double2half(0.01)
    #define kappa __double2half(1.8) 
    #define delta __double2half(0.02)
    #define aniso __double2half(4.0) 
    #define alpha __double2half(0.9) 
    #define gama __double2half(10.0) 
    #define teq __double2half(1.0) 
    #define theta0 __double2half(0.0) 
    #define dtime __double2half(1.0e-4)
    const int seed=10;
#endif
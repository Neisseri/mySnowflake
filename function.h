#ifndef FUNCTION_H
    #define FUNCTION_H
    #include"./paras.h"

    __global__ void kernel1_pure(highprecision *phi,highprecision* phi_lap,highprecision* tempr,highprecision* tempr_lap,highprecision* phidx,highprecision* phidy,highprecision* epsilon,highprecision* epsilon_deri){
        int unitindex_x = blockIdx.z % unitdimX;
        int unitindex_y = blockIdx.z / unitdimX;
        // 计算线程块在二维线程网络中的索引
        int x_offset = threadIdx.x;
        int x_start = unitindex_x * unitx;
        int x = x_start + x_offset;
        // 计算线程在当前线程块内的 x 轴方向的索引
        // 和在整个线程网格中的 x 轴方向的绝对索引
        int y_offset = threadIdx.y;
        int y = unitindex_y * unity + y_offset;

        int xs1 = x > 0 ? (x - 1) : (dimX - 1);
        int ys1 = y > 0 ? (y - 1) : (dimY - 1); // s1: (x - 1, y - 1)
        int xa1 = x < (dimX - 1) ? (x + 1) : 0;
        int ya1 = y < (dimY - 1) ? (y + 1) : 0; // a1: (x + 1, y + 1)
        int xs2 = (x -2) < 0 ? (x - 2) + dimX : (x - 2);
        int xa2 = (x + 2) > (dimX - 1) ? (x + 2 - dimX) : (x + 2);
        int ys2 = (y - 2) < 0 ? (y - 2 + dimY) : (y - 2); // s2: (x - 2, y - 2)
        int ya2 = (y + 2) > (dimY - 1) ? (y + 2 - dimY) :(y + 2); // a2: (x + 2, y + 2)
        
        highprecision(*phid)[dimX]=(highprecision(*)[dimX])phi;
        highprecision(*phi_lapd)[dimX]=(highprecision(*)[dimX])phi_lap;
        highprecision(*phidxd)[dimX]=(highprecision(*)[dimX])phidx;
        highprecision(*phidyd)[dimX]=(highprecision(*)[dimX])phidy;
        highprecision(*temprd)[dimX]=(highprecision(*)[dimX])tempr;
        highprecision(*tempr_lapd)[dimX]=(highprecision(*)[dimX])tempr_lap;
        highprecision(*epsilond)[dimX]=(highprecision(*)[dimX])epsilon;
        highprecision(*epsilon_derid)[dimX]=(highprecision(*)[dimX])epsilon_deri;
        // 将一维指针转换为二维数组
        
        phi_lapd[y][x] = 
            phid[y][xs1] / dxdy
            + phid[y][xs2] / dxdy
            + phid[y][xa1] / dxdy
            + phid[y][xa2] / dxdy
            + phid[ys1][x] / dxdy
            + phid[ys2][x] / dxdy
            + phid[ya1][x] / dxdy
            + phid[ya2][x] / dxdy
            - 8.0 * phid[y][x] / dxdy;

        tempr_lapd[y][x] = 
            temprd[y][xs1] / dxdy
            + temprd[y][xs2] / dxdy
            + temprd[y][xa1] / dxdy
            + temprd[y][xa2] / dxdy
            + temprd[ys1][x] / dxdy
            + temprd[ys2][x] / dxdy
            + temprd[ya1][x] / dxdy
            + temprd[ya2][x] / dxdy
            - 8.0 * temprd[y][x] / dxdy;

        phidxd[y][x] = (phid[y][xa1] - phid[y][xs1]) / (2.0 * dx);
        phidyd[y][x] = (phid[ya1][x] - phid[ys1][x]) / (2.0 * dy);

        highprecision theta = atan2(phidyd[y][x], phidxd[y][x]);
        epsilond[y][x] = epsilonb * (1.0 + delta * cos(aniso * (theta - theta0)));
        epsilon_derid[y][x] = -epsilonb * aniso * delta * sin(aniso * (theta - theta0));
    }

    __global__ void kernel2_pure(highprecision* phi,highprecision* phi_lap,highprecision* epsilon,highprecision *epsilon_deri,highprecision* phidx,highprecision* phidy,highprecision* tempr,highprecision* tempr_lap){
        int unitindex_x=blockIdx.z%unitdimX;
        int unitindex_y=blockIdx.z/unitdimX;
        int x_offset=threadIdx.x;
        int x_start=unitindex_x*unitx;
        int x=x_start+x_offset;
        int y_offset=threadIdx.y;
        int y=unitindex_y*unity+y_offset;
        int xs1=x>0?x-1:dimX-1;int ys1=y>0?y-1:dimY-1;
        int xa1=x<dimX-1?x+1:0;int ya1=y<dimY-1?y+1:0;
        highprecision(*phid)[dimX]=(highprecision(*)[dimX])phi;
        highprecision(*phi_lapd)[dimX]=(highprecision(*)[dimX])phi_lap;
        highprecision(*epsilond)[dimX]=(highprecision(*)[dimX])epsilon;
        highprecision(*epsilon_derid)[dimX]=(highprecision(*)[dimX])epsilon_deri;
        highprecision(*phidxd)[dimX]=(highprecision(*)[dimX])phidx;
        highprecision(*phidyd)[dimX]=(highprecision(*)[dimX])phidy;
        highprecision(*temprd)[dimX]=(highprecision(*)[dimX])tempr;
        highprecision(*tempr_lapd)[dimX]=(highprecision(*)[dimX])tempr_lap;
        highprecision phi_old=phid[y][x];
        highprecision term1=(epsilond[ya1][x]*epsilon_derid[ya1][x]*phidxd[ya1][x]-epsilond[ys1][x]*epsilon_derid[ys1][x]*phidxd[ys1][x])/(2.0*dy);
        highprecision term2=-(epsilond[y][xa1]*epsilon_derid[y][xa1]*phidyd[y][xa1]-epsilond[y][xs1]*epsilon_derid[y][xs1]*phidyd[y][xs1])/(2.0*dx);
        highprecision m=alpha/pi*atan(gama*(teq-temprd[y][x]));
        phid[y][x]=phid[y][x]+(dtime/tau)*(term1+term2+pow(epsilond[y][x],2)*phi_lapd[y][x]+phi_old*(1.0-phi_old)*(phi_old-0.5+m));
        temprd[y][x]=temprd[y][x]+dtime*tempr_lapd[y][x]+kappa*(phid[y][x]-phi_old);
    }    

#endif
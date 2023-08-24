#ifndef FUNCTION_H
    #define FUNCTION_H
    #include"./paras/para5.h"
    #if ((defined PURE)||(defined Motivation))
        __global__ void kernel1_pure(highprecision *phi,highprecision* phi_lap,highprecision* tempr,highprecision* tempr_lap,highprecision* phidx,highprecision* phidy,highprecision* epsilon,highprecision* epsilon_deri){
            int unitindex_x=blockIdx.z%unitdimX;
            int unitindex_y=blockIdx.z/unitdimX;
            // 计算线程块在二维线程网络中的索引
            int x_offset=threadIdx.x ;
            int x_start=unitindex_x*unitx;
            int x=x_start+x_offset;
            // 计算线程在当前线程块内的 x 轴方向的索引
            // 和在整个线程网格中的 x 轴方向的绝对索引
            int y_offset=threadIdx.y;
            int y=unitindex_y*unity+y_offset;

            int xs1=x>0?x-1:dimX-1;int ys1=y>0?y-1:dimY-1; // s1 : (x - 1, y - 1)
            int xa1=x<dimX-1?x+1:0;int ya1=y<dimY-1?y+1:0; // a1: (x + 1, y + 1)
            int xs2=x-2<0?x-2+dimX:x-2;
            int xa2=x+2>dimX-1?x+2-dimX:x+2;
            int ys2=y-2<0?y-2+dimY:y-2; // s2: (x - 2, y - 2)
            int ya2=y+2>dimY-1?y+2-dimY:y+2; // a2: (x + 2, y + 2)
            highprecision(*phid)[dimX]=(highprecision(*)[dimX])phi;
            highprecision(*phi_lapd)[dimX]=(highprecision(*)[dimX])phi_lap;
            highprecision(*phidxd)[dimX]=(highprecision(*)[dimX])phidx;
            highprecision(*phidyd)[dimX]=(highprecision(*)[dimX])phidy;
            highprecision(*temprd)[dimX]=(highprecision(*)[dimX])tempr;
            highprecision(*tempr_lapd)[dimX]=(highprecision(*)[dimX])tempr_lap;
            highprecision(*epsilond)[dimX]=(highprecision(*)[dimX])epsilon;
            highprecision(*epsilon_derid)[dimX]=(highprecision(*)[dimX])epsilon_deri;
            // 将一维指针转换为二维数组
            phi_lapd[y][x]=phid[y][xs1]/dxdy+phid[y][xs2]/dxdy+phid[y][xa1]/dxdy+phid[y][xa2]/dxdy+phid[ys1][x]/dxdy+phid[ys2][x]/dxdy+phid[ya1][x]/dxdy+phid[ya2][x]/dxdy-8.0*phid[y][x]/dxdy;
            tempr_lapd[y][x]=temprd[y][xs1]/dxdy+temprd[y][xs2]/dxdy+temprd[y][xa1]/dxdy+temprd[y][xa2]/dxdy+temprd[ys1][x]/dxdy+temprd[ys2][x]/dxdy+temprd[ya1][x]/dxdy+temprd[ya2][x]/dxdy-8.0*temprd[y][x]/dxdy;

            phidxd[y][x]=(phid[y][xa1]-phid[y][xs1])/(2.0*dx);
            phidyd[y][x]=(phid[ya1][x]-phid[ys1][x])/(2.0*dy);
            highprecision theta=atan2(phidyd[y][x],phidxd[y][x]);
            epsilond[y][x]=epsilonb*(1.0+delta*cos(aniso*(theta-theta0)));
            epsilon_derid[y][x]=-epsilonb*aniso*delta*sin(aniso*(theta-theta0)); // UNKNOWN
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
    #if ((defined AMSTENCIL)||(defined GRAM1||(defined GRAM2)))
    __global__ void kernel1_mix(highprecision* phi,highprecision *phi_last,highprecision* phi_lap,highprecision* epsilon,highprecision *epsilon_deri,highprecision* phidx,highprecision* phidy,highprecision* tempr,highprecision* tempr_lap,lowprecision *hphi,lowprecision* hphi_lap,lowprecision* htempr,lowprecision* htempr_lap,lowprecision* hphidx,lowprecision* hphidy,lowprecision* hepsilon,lowprecision* hepsilon_deri,lowprecision hdxdy,lowprecision htheta0,lowprecision haniso,lowprecision hone,lowprecision hdxm2,lowprecision hdym2,lowprecision hdelta,lowprecision hepsilonb,lowprecision height,int* type_curr,int i){
        int unitindex_x=blockIdx.z%unitdimX;
        int unitindex_y=blockIdx.z/unitdimX;
        int type;
        #ifdef AMSTENCIL
        int(*type_currd)[unitdimX]=(int(*)[unitdimX])type_curr;
        type=type_currd[unitindex_y][unitindex_x];
        if(type==1&&blockIdx.x==1)return;
        #endif
        #ifdef GRAM1
            if(blockIdx.z<unitNums*ratio)type=1;
            else type=2;
        #endif
        #ifdef GRAM2
            if(blockIdx.z%100<100*ratio)type=1;
            else type=2;
        #endif
        int unitindex_ys1=unitindex_y==0?unitdimY-1:unitindex_y-1;
        int unitindex_ya1=unitindex_y==unitdimY-1?0:unitindex_y+1;
        int x_offset=threadIdx.x;
        int x_start=unitindex_x*uxd2;
        int y_offset=threadIdx.y;
        int y=unitindex_y*unity+y_offset;
        int ys1=y>0?y-1:dimY-1;
        int ya1=y<dimY-1?y+1:0;
        int ys2=(y-2)>=0?y-2:y-2+dimY;
        int ya2=(y+2)<dimY?y+2:y+2-dimY;
        lowprecision(*hphi_lapd)[lowprecison_dimX]=(lowprecision(*)[lowprecison_dimX])hphi_lap;
        lowprecision(*htempr_lapd)[lowprecison_dimX]=(lowprecision(*)[lowprecison_dimX])htempr_lap;
        lowprecision(*hphidxd)[lowprecison_dimX]=(lowprecision(*)[lowprecison_dimX])hphidx;
        lowprecision(*hphidyd)[lowprecison_dimX]=(lowprecision(*)[lowprecison_dimX])hphidy;
        lowprecision(*hepsilond)[lowprecison_dimX]=(lowprecision(*)[lowprecison_dimX])hepsilon;
        lowprecision(*hepsilon_derid)[lowprecison_dimX]=(lowprecision(*)[lowprecison_dimX])hepsilon_deri;
        lowprecision(*hphid)[lowprecison_dimX]=(lowprecision(*)[lowprecison_dimX])hphi;
        lowprecision(*htemprd)[lowprecison_dimX]=(lowprecision(*)[lowprecison_dimX])htempr;
        highprecision(*phi_lapd)[dimX]=(highprecision(*)[dimX])phi_lap;
        highprecision(*phidxd)[dimX]=(highprecision(*)[dimX])phidx;
        highprecision(*phidyd)[dimX]=(highprecision(*)[dimX])phidy;
        highprecision(*tempr_lapd)[dimX]=(highprecision(*)[dimX])tempr_lap;
        highprecision(*epsilond)[dimX]=(highprecision(*)[dimX])epsilon;
        highprecision(*epsilon_derid)[dimX]=(highprecision(*)[dimX])epsilon_deri;
        highprecision(*phid)[dimX]=(highprecision(*)[dimX])phi;
        highprecision(*temprd)[dimX]=(highprecision(*)[dimX])tempr;
        #ifdef monitor_conversion_dependent
            #ifdef Monitor2
                if(i%10==0){
                    highprecision(*phi_lastd)[dimX]=(highprecision(*)[dimX])phi_last;
                    phi_lastd[y][x_start*2+x_offset+uxd2*blockIdx.x]=phid[y][x_start*2+x_offset+uxd2*blockIdx.x];
                }
            #endif
        #endif
        if(type==1){
            int unitindex_xs1=unitindex_x==0?unitdimX-1:unitindex_x-1;
            int unitindex_xa1=unitindex_x==unitdimX-1?0:unitindex_x+1;
            //
            #ifdef AMSTENCIL
                int x=x_start+x_offset;
            #else
                x_offset=x_offset+blockIdx.x*uxd2;
                int x=x_start*2+x_offset;
            #endif
            int xs1=x>0?x-1:lowprecison_dimX-1;
            int xa1=x<lowprecison_dimX-1?x+1:0;

            #ifdef AMSTENCIL
                lowprecision phi_xs1= __halves2half2(__high2half(hphid[y][xs1]),__low2half(hphid[y][x]));
                lowprecision phi_xs2=hphid[y][xs1];
                lowprecision phi_xa1=__halves2half2(__high2half(hphid[y][x]),__low2half(hphid[y][xa1]));
                lowprecision phi_xa2=hphid[y][xa1];
                lowprecision phi_ys1=hphid[ys1][x];
                lowprecision phi_ys2=hphid[ys2][x];
                lowprecision phi_ya1=hphid[ya1][x];
                lowprecision phi_ya2=hphid[ya2][x];

                lowprecision tempr_xs1= __halves2half2(__high2half(htemprd[y][xs1]),__low2half(htemprd[y][x]));
                lowprecision tempr_xs2=htemprd[y][xs1];
                lowprecision tempr_xa1=__halves2half2(__high2half(htemprd[y][x]),__low2half(htemprd[y][xa1]));
                lowprecision tempr_xa2=htemprd[y][xa1];
                lowprecision tempr_ys1=htemprd[ys1][x];
                lowprecision tempr_ys2=htemprd[ys2][x];
                lowprecision tempr_ya1=htemprd[ya1][x];
                lowprecision tempr_ya2=htemprd[ya2][x];
                if(x_offset==0&&type_currd[unitindex_y][unitindex_xs1]==2){
                    phi_xs1=__halves2half2(__float2half(phid[y][2*xs1+1]),__high2half(phi_xs1));
                    tempr_xs1= __halves2half2(__float2half(temprd[y][2*xs1+1]),__high2half(tempr_xs1));
                    phi_xs2=__halves2half2(__float2half(phid[y][2*xs1]),__float2half(phid[y][2*xs1+1]));
                    tempr_xs2= __halves2half2(__float2half(temprd[y][2*xs1]),__float2half(temprd[y][2*xs1+1]));
                }
                else if(x_offset==1&&type_currd[unitindex_y][unitindex_xs1]==2){
                    phi_xs2=__halves2half2(__float2half(phid[y][2*xs1]),__float2half(phid[y][2*xs1+1]));
                    tempr_xs2= __halves2half2(__float2half(temprd[y][2*xs1]),__float2half(temprd[y][2*xs1+1]));
                }
                else if(x_offset==uxd2s1&&type_currd[unitindex_y][unitindex_xa1]==2){
                    phi_xa1=__halves2half2(__low2half(phi_xa1),__float2half(phid[y][2*xa1]));
                    tempr_xa1=__halves2half2(__low2half(tempr_xa1),__float2half(temprd[y][2*xa1]));
                    phi_xa2=__halves2half2(__float2half(phid[y][2*xa1]),__float2half(phid[y][2*xa1+1]));
                    tempr_xa2= __halves2half2(__float2half(temprd[y][2*xa1]),__float2half(temprd[y][2*xa1+1]));
                }
                else if(x_offset==uxd2s1-1&&type_currd[unitindex_y][unitindex_xa1]==2){
                    phi_xa2=__halves2half2(__float2half(phid[y][2*xa1]),__float2half(phid[y][2*xa1+1]));
                    tempr_xa2= __halves2half2(__float2half(temprd[y][2*xa1]),__float2half(temprd[y][2*xa1+1]));  
                }

                if(y_offset==0 and type_currd[unitindex_ys1][unitindex_x]==2){
                    phi_ys1=__floats2half2_rn(phid[ys1][x*2],phid[ys1][x*2+1]);
                    tempr_ys1=__floats2half2_rn(temprd[ys1][x*2],temprd[ys1][x*2+1]);
                    phi_ys2=__floats2half2_rn(phid[ys2][x*2],phid[ys2][x*2+1]);
                    tempr_ys2=__floats2half2_rn(temprd[ys2][x*2],temprd[ys2][x*2+1]);
                }
                else if(y_offset==1 and type_currd[unitindex_ys1][unitindex_x]==2){
                    phi_ys2=__floats2half2_rn(phid[ys2][x*2],phid[ys2][x*2+1]);
                    tempr_ys2=__floats2half2_rn(temprd[ys2][x*2],temprd[ys2][x*2+1]);
                }
                else if(y_offset==uys1 and type_currd[unitindex_ya1][unitindex_x]==2){
                    phi_ya1=__floats2half2_rn(phid[ya1][x*2],phid[ya1][x*2+1]);
                    tempr_ya1=__floats2half2_rn(temprd[ya1][x*2],temprd[ya1][x*2+1]);
                    phi_ya2=__floats2half2_rn(phid[ya2][x*2],phid[ya2][x*2+1]);
                    tempr_ya2=__floats2half2_rn(temprd[ya2][x*2],temprd[ya2][x*2+1]);
                }
                else if(y_offset==uys1-1 and type_currd[unitindex_ya1][unitindex_x]==2){
                    phi_ya2=__floats2half2_rn(phid[ya2][x*2],phid[ya2][x*2+1]);
                    tempr_ya2=__floats2half2_rn(temprd[ya2][x*2],temprd[ya2][x*2+1]);
                }
            #endif
            #ifdef GRAM1
                int xs2=(x-2)>=0?x-2:x-2+lowprecison_dimX;
                int xa2=(x+2)<lowprecison_dimX?x+2:x+2-lowprecison_dimX;
                lowprecision phi_xs1=(x_offset<1&&(unitindex_y*unitdimX+unitindex_xs1)>=unitNums*ratio)?__float2half(phid[y][xs1]):hphid[y][xs1];
                lowprecision phi_xs2=(x_offset<2&&(unitindex_y*unitdimX+unitindex_xs1)>=unitNums*ratio)?__float2half(phid[y][xs2]):hphid[y][xs2];
                lowprecision phi_xa1=(x_offset+1>uxs1&&(unitindex_y*unitdimX+unitindex_xa1)>=unitNums*ratio)?__float2half(phid[y][xa1]):hphid[y][xa1];
                lowprecision phi_xa2=(x_offset+2>uxs1&&(unitindex_y*unitdimX+unitindex_xa1)>=unitNums*ratio)?__float2half(phid[y][xa2]):hphid[y][xa2];
                lowprecision phi_ys1=(y_offset<1&&(unitindex_ys1*unitdimX+unitindex_x)>=unitNums*ratio)?__float2half(phid[ys1][x]):hphid[ys1][x];
                lowprecision phi_ys2=(y_offset<2&&(unitindex_ys1*unitdimX+unitindex_x)>=unitNums*ratio)?__float2half(phid[ys2][x]):hphid[ys2][x];
                lowprecision phi_ya1=(y_offset+1>uys1&&(unitindex_ya1*unitdimX+unitindex_x)>=unitNums*ratio)?__float2half(phid[ya1][x]):hphid[ya1][x];
                lowprecision phi_ya2=(y_offset+2>uys1&&(unitindex_ya1*unitdimX+unitindex_x)>=unitNums*ratio)?__float2half(phid[ya2][x]):hphid[ya2][x];

                lowprecision tempr_xs1=(x_offset<1&&(unitindex_y*unitdimX+unitindex_xs1)>=unitNums*ratio)?__float2half(temprd[y][xs1]):htemprd[y][xs1];
                lowprecision tempr_xs2=(x_offset<2&&(unitindex_y*unitdimX+unitindex_xs1)>=unitNums*ratio)?__float2half(temprd[y][xs2]):htemprd[y][xs2];
                lowprecision tempr_xa1=(x_offset+1>uxs1&&(unitindex_y*unitdimX+unitindex_xa1)>=unitNums*ratio)?__float2half(temprd[y][xa1]):htemprd[y][xa1];
                lowprecision tempr_xa2=(x_offset+2>uxs1&&(unitindex_y*unitdimX+unitindex_xa1)>=unitNums*ratio)?__float2half(temprd[y][xa2]):htemprd[y][xa2];
                lowprecision tempr_ys1=(y_offset<1&&(unitindex_ys1*unitdimX+unitindex_x)>=unitNums*ratio)?__float2half(temprd[ys1][x]):htemprd[ys1][x];
                lowprecision tempr_ys2=(y_offset<2&&(unitindex_ys1*unitdimX+unitindex_x)>=unitNums*ratio)?__float2half(temprd[ys2][x]):htemprd[ys2][x];
                lowprecision tempr_ya1=(y_offset+1>uys1&&(unitindex_ya1*unitdimX+unitindex_x)>=unitNums*ratio)?__float2half(temprd[ya1][x]):htemprd[ya1][x];
                lowprecision tempr_ya2=(y_offset+2>uys1&&(unitindex_ya1*unitdimX+unitindex_x)>=unitNums*ratio)?__float2half(temprd[ya2][x]):htemprd[ya2][x];
            #endif
            #ifdef GRAM2
                int xs2=(x-2)>=0?x-2:x-2+lowprecison_dimX;
                int xa2=(x+2)<lowprecison_dimX?x+2:x+2-lowprecison_dimX;
                lowprecision phi_xs1=(x_offset<1&&(unitindex_y*unitdimX+unitindex_xs1)%100>=100*ratio)?__float2half(phid[y][xs1]):hphid[y][xs1];
                lowprecision phi_xs2=(x_offset<2&&(unitindex_y*unitdimX+unitindex_xs1)%100>=100*ratio)?__float2half(phid[y][xs2]):hphid[y][xs2];
                lowprecision phi_xa1=(x_offset+1>uxs1&&(unitindex_y*unitdimX+unitindex_xa1)%100>=100*ratio)?__float2half(phid[y][xa1]):hphid[y][xa1];
                lowprecision phi_xa2=(x_offset+2>uxs1&&(unitindex_y*unitdimX+unitindex_xa1)%100>=100*ratio)?__float2half(phid[y][xa2]):hphid[y][xa2];
                lowprecision phi_ys1=(y_offset<1&&(unitindex_ys1*unitdimX+unitindex_x)%100>=100*ratio)?__float2half(phid[ys1][x]):hphid[ys1][x];
                lowprecision phi_ys2=(y_offset<2&&(unitindex_ys1*unitdimX+unitindex_x)%100>=100*ratio)?__float2half(phid[ys2][x]):hphid[ys2][x];
                lowprecision phi_ya1=(y_offset+1>uys1&&(unitindex_ya1*unitdimX+unitindex_x)%100>=100*ratio)?__float2half(phid[ya1][x]):hphid[ya1][x];
                lowprecision phi_ya2=(y_offset+2>uys1&&(unitindex_ya1*unitdimX+unitindex_x)%100>=100*ratio)?__float2half(phid[ya2][x]):hphid[ya2][x];

                lowprecision tempr_xs1=(x_offset<1&&(unitindex_y*unitdimX+unitindex_xs1)%100>=100*ratio)?__float2half(temprd[y][xs1]):htemprd[y][xs1];
                lowprecision tempr_xs2=(x_offset<2&&(unitindex_y*unitdimX+unitindex_xs1)%100>=100*ratio)?__float2half(temprd[y][xs2]):htemprd[y][xs2];
                lowprecision tempr_xa1=(x_offset+1>uxs1&&(unitindex_y*unitdimX+unitindex_xa1)%100>=100*ratio)?__float2half(temprd[y][xa1]):htemprd[y][xa1];
                lowprecision tempr_xa2=(x_offset+2>uxs1&&(unitindex_y*unitdimX+unitindex_xa1)%100>=100*ratio)?__float2half(temprd[y][xa2]):htemprd[y][xa2];
                lowprecision tempr_ys1=(y_offset<1&&(unitindex_ys1*unitdimX+unitindex_x)%100>=100*ratio)?__float2half(temprd[ys1][x]):htemprd[ys1][x];
                lowprecision tempr_ys2=(y_offset<2&&(unitindex_ys1*unitdimX+unitindex_x)%100>=100*ratio)?__float2half(temprd[ys2][x]):htemprd[ys2][x];
                lowprecision tempr_ya1=(y_offset+1>uys1&&(unitindex_ya1*unitdimX+unitindex_x)%100>=100*ratio)?__float2half(temprd[ya1][x]):htemprd[ya1][x];
                lowprecision tempr_ya2=(y_offset+2>uys1&&(unitindex_ya1*unitdimX+unitindex_x)%100>=100*ratio)?__float2half(temprd[ya2][x]):htemprd[ya2][x];
            #endif
            hphi_lapd[y][x]=phi_xs1/hdxdy+phi_xs2/hdxdy+phi_xa1/hdxdy+phi_xa2/hdxdy+phi_ys1/hdxdy+phi_ys2/hdxdy+phi_ya1/hdxdy+phi_ya2/hdxdy-height*hphid[y][x]/hdxdy;
            htempr_lapd[y][x]=tempr_xs1/hdxdy+tempr_xs2/hdxdy+tempr_xa1/hdxdy+tempr_xa2/hdxdy+tempr_ys1/hdxdy+tempr_ys2/hdxdy+tempr_ya1/hdxdy+tempr_ya2/hdxdy-height*htemprd[y][x]/hdxdy;
            hphidxd[y][x]=(phi_xa1-phi_xs1)/hdxm2;
            hphidyd[y][x]=(phi_ya1-phi_ys1)/hdym2;
            #ifdef AMSTENCIL
                lowprecision theta=__floats2half2_rn(atan2(__low2float(hphidyd[y][x]),__low2float(hphidxd[y][x])),atan2(__high2float(hphidyd[y][x]),__high2float(hphidxd[y][x])));
                lowprecision tmp=haniso*(theta-htheta0);
                lowprecision tmp1=h2cos(tmp);
                lowprecision tmp2=h2sin(tmp);
            #else
                lowprecision theta=__float2half(atan2(__half2float(hphidyd[y][x]),__half2float(hphidxd[y][x])));
                lowprecision tmp=haniso*(theta-htheta0);
                lowprecision tmp1=hcos(tmp);
                lowprecision tmp2=hsin(tmp);
            #endif
            hepsilond[y][x]=hepsilonb*(hone+hdelta*tmp1);
            hepsilon_derid[y][x]=-hepsilonb*haniso*hdelta*tmp2;
            #ifdef monitor_conversion_dependent
                #ifdef AMSTENCIL
                    if(y_offset==0||y_offset==uys1){
                        phidxd[y][x*2]=__low2float(hphidxd[y][x]);
                        phidxd[y][x*2+1]=__high2float(hphidxd[y][x]);
                        phidyd[y][x*2]=__low2float(hphidyd[y][x]);
                        phidyd[y][x*2+1]=__high2float(hphidyd[y][x]);
                        epsilond[y][x*2]=__low2float(hepsilond[y][x]);
                        epsilond[y][x*2+1]=__high2float(hepsilond[y][x]);
                        epsilon_derid[y][x*2]=__low2float(hepsilon_derid[y][x]);
                        epsilon_derid[y][x*2+1]=__high2float(hepsilon_derid[y][x]);
                    }
                    else if(x_offset==0){
                        phidxd[y][x*2]=__low2float(hphidxd[y][x]);
                        phidyd[y][x*2]=__low2float(hphidyd[y][x]);
                        epsilond[y][x*2]=__low2float(hepsilond[y][x]);
                        epsilon_derid[y][x*2]=__low2float(hepsilon_derid[y][x]);
                    }
                    else if(x_offset==uxd2s1){
                        phidxd[y][x*2+1]=__high2float(hphidxd[y][x]);
                        phidyd[y][x*2+1]=__high2float(hphidyd[y][x]);
                        epsilond[y][x*2+1]=__high2float(hepsilond[y][x]);
                        epsilon_derid[y][x*2+1]=__high2float(hepsilon_derid[y][x]);
                    }

                #endif
            #endif
            // #ifdef PRINT_INFO
            //     #ifdef AMSTENCIL
            //         if(x==321/2 and y==254 and i==110){

            //             // printf("amst philap:%10e,phixs1:%10e,phixs2:%10e,phixa1:%10e,phixa2:%10e,phiys1:%10e,phiys2:%10e,phiya1:%10e,phiya2:%10e,phi:%10e\n",__low2float(hphi_lapd[y][x]),__low2float(phi_xs1),__low2float(phi_xs2),__low2float(phi_xa1),__low2float(phi_xa2),__low2float(phi_ys1),__low2float(phi_ys2),__low2float(phi_ya1),__low2float(phi_ya2),__low2float(hphid[y][x]));
            //             // printf("amst temprlap:%10e,temprxs1:%10e,temprxs2:%10e,temprxa1:%10e,temprxa2:%10e,temprys1:%10e,temprys2:%10e,temprya1:%10e,temprya2:%10e,tempr:%10e\n",__low2float(htempr_lapd[y][x]),__low2float(tempr_xs1),__low2float(tempr_xs2),__low2float(tempr_xa1),__low2float(tempr_xa2),__low2float(tempr_ys1),__low2float(tempr_ys2),__low2float(tempr_ya1),__low2float(tempr_ya2),__low2float(htemprd[y][x]));
                        
            //             // printf("amst philap:%10e,phixs1:%10e,phixs2:%10e,phixa1:%10e,phixa2:%10e,phiys1:%10e,phiys2:%10e,phiya1:%10e,phiya2:%10e,phi:%10e\n",__high2float(hphi_lapd[y][x]),__high2float(phi_xs1),__high2float(phi_xs2),__high2float(phi_xa1),__high2float(phi_xa2),__high2float(phi_ys1),__high2float(phi_ys2),__high2float(phi_ya1),__high2float(phi_ya2),__high2float(hphid[y][x]));
            //             printf("amst temprlap:%10e,temprxs1:%10e,temprxs2:%10e,temprxa1:%10e,temprxa2:%10e,temprys1:%10e,temprys2:%10e,temprya1:%10e,temprya2:%10e,tempr:%10e\n",__high2float(htempr_lapd[y][x]),__high2float(tempr_xs1),__high2float(tempr_xs2),__high2float(tempr_xa1),__high2float(tempr_xa2),__high2float(tempr_ys1),__high2float(tempr_ys2),__high2float(tempr_ya1),__high2float(tempr_ya2),__high2float(htemprd[y][x]));


            //         }
            //     #else
            //         if(x==321 and y==254 and i==110){
            //             // printf("gram philap:%10e,phixs1:%10e,phixs2:%10e,phixa1:%10e,phixa2:%10e,phiys1:%10e,phiys2:%10e,phiya1:%10e,phiya2:%10e,phi:%10e\n",(float)hphi_lapd[y][x],(float)phi_xs1,(float)phi_xs2,(float)phi_xa1,(float)phi_xa2,(float)phi_ys1,(float)phi_ys2,(float)phi_ya1,(float)phi_ya2,(float)hphid[y][x]);
            //             printf("gram temprlap:%10e,temprxs1:%10e,temprxs2:%10e,temprxa1:%10e,temprxa2:%10e,temprys1:%10e,temprys2:%10e,temprya1:%10e,temprya2:%10e,tempr:%10e\n",(float)htempr_lapd[y][x],(float)tempr_xs1,(float)tempr_xs2,(float)tempr_xa1,(float)tempr_xa2,(float)tempr_ys1,(float)tempr_ys2,(float)tempr_ya1,(float)tempr_ya2,(float)htemprd[y][x]);
            //         }
            //     #endif
            // #endif
        }
        else{
            x_offset=x_offset+blockIdx.x*uxd2;
            int x=x_start*2+x_offset;
            int xs1=x>0?x-1:dimX-1;
            int xa1=x<dimX-1?x+1:0;
            int xs2=x-2<0?x-2+dimX:x-2;
            int xa2=x+2>dimX-1?x+2-dimX:x+2;
            #ifdef AMSTENCIL
                phi_lapd[y][x]=phid[y][xs1]/dxdy+phid[y][xs2]/dxdy+phid[y][xa1]/dxdy+phid[y][xa2]/dxdy+phid[ys1][x]/dxdy+phid[ys2][x]/dxdy+phid[ya1][x]/dxdy+phid[ya2][x]/dxdy-8.0*phid[y][x]/dxdy;
                tempr_lapd[y][x]=temprd[y][xs1]/dxdy+temprd[y][xs2]/dxdy+temprd[y][xa1]/dxdy+temprd[y][xa2]/dxdy+temprd[ys1][x]/dxdy+temprd[ys2][x]/dxdy+temprd[ya1][x]/dxdy+temprd[ya2][x]/dxdy-8.0*temprd[y][x]/dxdy;
                phidxd[y][x]=(phid[y][xa1]-phid[y][xs1])/(2.0*dx);
                phidyd[y][x]=(phid[ya1][x]-phid[ys1][x])/(2.0*dy);
            #endif
            #ifdef GRAM1
                int unitindex_xs1=unitindex_x==0?unitdimX-1:unitindex_x-1;
                int unitindex_xa1=unitindex_x==unitdimX-1?0:unitindex_x+1;
                highprecision phi_xs1=(x_offset<1&&(unitindex_y*unitdimX+unitindex_xs1)<unitNums*ratio)?__half2float(hphid[y][xs1]):phid[y][xs1];
                highprecision phi_xs2=(x_offset<2&&(unitindex_y*unitdimX+unitindex_xs1)<unitNums*ratio)?__half2float(hphid[y][xs2]):phid[y][xs2];
                highprecision phi_xa1=(x_offset+1>uxs1&&(unitindex_y*unitdimX+unitindex_xa1)<unitNums*ratio)?__half2float(hphid[y][xa1]):phid[y][xa1];
                highprecision phi_xa2=(x_offset+2>uxs1&&(unitindex_y*unitdimX+unitindex_xa1)<unitNums*ratio)?__half2float(hphid[y][xa2]):phid[y][xa2];
                highprecision phi_ys1=(y_offset<1&&(unitindex_ys1*unitdimX+unitindex_x)<unitNums*ratio)?__half2float(hphid[ys1][x]):phid[ys1][x];
                highprecision phi_ys2=(y_offset<2&&(unitindex_ys1*unitdimX+unitindex_x)<unitNums*ratio)?__half2float(hphid[ys2][x]):phid[ys2][x];
                highprecision phi_ya1=(y_offset+1>uys1&&(unitindex_ya1*unitdimX+unitindex_x)<unitNums*ratio)?__half2float(hphid[ya1][x]):phid[ya1][x];
                highprecision phi_ya2=(y_offset+2>uys1&&(unitindex_ya1*unitdimX+unitindex_x)<unitNums*ratio)?__half2float(hphid[ya2][x]):phid[ya2][x];
                phi_lapd[y][x]=phi_xs1/dxdy+phi_xs2/dxdy+phi_xa1/dxdy+phi_xa2/dxdy+phi_ys1/dxdy+phi_ys2/dxdy+phi_ya1/dxdy+phi_ya2/dxdy-8.0*phid[y][x]/dxdy;

                highprecision tempr_xs1=(x_offset<1&&(unitindex_y*unitdimX+unitindex_xs1)<unitNums*ratio)?__half2float(htemprd[y][xs1]):temprd[y][xs1];
                highprecision tempr_xs2=(x_offset<2&&(unitindex_y*unitdimX+unitindex_xs1)<unitNums*ratio)?__half2float(htemprd[y][xs2]):temprd[y][xs2];
                highprecision tempr_xa1=(x_offset+1>uxs1&&(unitindex_y*unitdimX+unitindex_xa1)<unitNums*ratio)?__half2float(htemprd[y][xa1]):temprd[y][xa1];
                highprecision tempr_xa2=(x_offset+2>uxs1&&(unitindex_y*unitdimX+unitindex_xa1)<unitNums*ratio)?__half2float(htemprd[y][xa2]):temprd[y][xa2];
                highprecision tempr_ys1=(y_offset<1&&(unitindex_ys1*unitdimX+unitindex_x)<unitNums*ratio)?__half2float(htemprd[ys1][x]):temprd[ys1][x];
                highprecision tempr_ys2=(y_offset<2&&(unitindex_ys1*unitdimX+unitindex_x)<unitNums*ratio)?__half2float(htemprd[ys2][x]):temprd[ys2][x];
                highprecision tempr_ya1=(y_offset+1>uys1&&(unitindex_ya1*unitdimX+unitindex_x)<unitNums*ratio)?__half2float(htemprd[ya1][x]):temprd[ya1][x];
                highprecision tempr_ya2=(y_offset+2>uys1&&(unitindex_ya1*unitdimX+unitindex_x)<unitNums*ratio)?__half2float(htemprd[ya2][x]):temprd[ya2][x];
                tempr_lapd[y][x]=tempr_xs1/dxdy+tempr_xs2/dxdy+tempr_xa1/dxdy+tempr_xa2/dxdy+tempr_ys1/dxdy+tempr_ys2/dxdy+tempr_ya1/dxdy+tempr_ya2/dxdy-8.0*temprd[y][x]/dxdy;
                phidxd[y][x]=(phi_xa1-phi_xs1)/(2.0*dx);
                phidyd[y][x]=(phi_ya1-phi_ys1)/(2.0*dy);
            #endif
            #ifdef GRAM2
                int unitindex_xs1=unitindex_x==0?unitdimX-1:unitindex_x-1;
                int unitindex_xa1=unitindex_x==unitdimX-1?0:unitindex_x+1;
                highprecision phi_xs1=(x_offset<1&&(unitindex_y*unitdimX+unitindex_xs1)%100<100*ratio)?__half2float(hphid[y][xs1]):phid[y][xs1];
                highprecision phi_xs2=(x_offset<2&&(unitindex_y*unitdimX+unitindex_xs1)%100<100*ratio)?__half2float(hphid[y][xs2]):phid[y][xs2];
                highprecision phi_xa1=(x_offset+1>uxs1&&(unitindex_y*unitdimX+unitindex_xa1)%100<100*ratio)?__half2float(hphid[y][xa1]):phid[y][xa1];
                highprecision phi_xa2=(x_offset+2>uxs1&&(unitindex_y*unitdimX+unitindex_xa1)%100<100*ratio)?__half2float(hphid[y][xa2]):phid[y][xa2];
                highprecision phi_ys1=(y_offset<1&&(unitindex_ys1*unitdimX+unitindex_x)%100<100*ratio)?__half2float(hphid[ys1][x]):phid[ys1][x];
                highprecision phi_ys2=(y_offset<2&&(unitindex_ys1*unitdimX+unitindex_x)%100<100*ratio)?__half2float(hphid[ys2][x]):phid[ys2][x];
                highprecision phi_ya1=(y_offset+1>uys1&&(unitindex_ya1*unitdimX+unitindex_x)%100<100*ratio)?__half2float(hphid[ya1][x]):phid[ya1][x];
                highprecision phi_ya2=(y_offset+2>uys1&&(unitindex_ya1*unitdimX+unitindex_x)%100<100*ratio)?__half2float(hphid[ya2][x]):phid[ya2][x];
                phi_lapd[y][x]=phi_xs1/dxdy+phi_xs2/dxdy+phi_xa1/dxdy+phi_xa2/dxdy+phi_ys1/dxdy+phi_ys2/dxdy+phi_ya1/dxdy+phi_ya2/dxdy-8.0*phid[y][x]/dxdy;

                highprecision tempr_xs1=(x_offset<1&&(unitindex_y*unitdimX+unitindex_xs1)%100<100*ratio)?__half2float(htemprd[y][xs1]):temprd[y][xs1];
                highprecision tempr_xs2=(x_offset<2&&(unitindex_y*unitdimX+unitindex_xs1)%100<100*ratio)?__half2float(htemprd[y][xs2]):temprd[y][xs2];
                highprecision tempr_xa1=(x_offset+1>uxs1&&(unitindex_y*unitdimX+unitindex_xa1)%100<100*ratio)?__half2float(htemprd[y][xa1]):temprd[y][xa1];
                highprecision tempr_xa2=(x_offset+2>uxs1&&(unitindex_y*unitdimX+unitindex_xa1)%100<100*ratio)?__half2float(htemprd[y][xa2]):temprd[y][xa2];
                highprecision tempr_ys1=(y_offset<1&&(unitindex_ys1*unitdimX+unitindex_x)%100<100*ratio)?__half2float(htemprd[ys1][x]):temprd[ys1][x];
                highprecision tempr_ys2=(y_offset<2&&(unitindex_ys1*unitdimX+unitindex_x)%100<100*ratio)?__half2float(htemprd[ys2][x]):temprd[ys2][x];
                highprecision tempr_ya1=(y_offset+1>uys1&&(unitindex_ya1*unitdimX+unitindex_x)%100<100*ratio)?__half2float(htemprd[ya1][x]):temprd[ya1][x];
                highprecision tempr_ya2=(y_offset+2>uys1&&(unitindex_ya1*unitdimX+unitindex_x)%100<100*ratio)?__half2float(htemprd[ya2][x]):temprd[ya2][x];
                tempr_lapd[y][x]=tempr_xs1/dxdy+tempr_xs2/dxdy+tempr_xa1/dxdy+tempr_xa2/dxdy+tempr_ys1/dxdy+tempr_ys2/dxdy+tempr_ya1/dxdy+tempr_ya2/dxdy-8.0*temprd[y][x]/dxdy;
                phidxd[y][x]=(phi_xa1-phi_xs1)/(2.0*dx);
                phidyd[y][x]=(phi_ya1-phi_ys1)/(2.0*dy);
            #endif

            highprecision theta=atan2(phidyd[y][x],phidxd[y][x]);
            epsilond[y][x]=epsilonb*(1.0+delta*cos(aniso*(theta-theta0)));
            epsilon_derid[y][x]=-epsilonb*aniso*delta*sin(aniso*(theta-theta0));
            // #ifdef PRINT_INFO
            //     #ifdef AMSTENCIL
            //         if(x==265 and y==272 and i==2){
            //             // printf("amst tempr%10e,tempr_lap:%10e,xs1:%10e,xs2:%10e,ys1:%10e,ys2:%10e\n",temprd[y][x],tempr_lapd[y][x],temprd[y][xs1],temprd[y][xs2],temprd[ys1][x],temprd[ys2][x]);
            //             printf("amst phi%10e,phi_lap:%10e,xs1:%10e,xs2:%10e,ys1:%10e,ys2:%10e\n",phid[y][x],phi_lapd[y][x],phid[y][xs1],temprd[y][xs2],phid[ys1][x],phid[ys2][x]);

            //         }
            //     #else
            //         if(x==265 and y==272 and i==2){
            //             // printf("gram tempr%10e,tempr_lap:%10e,xs1:%10e,xs2:%10e,ys1:%10e,ys2:%10e\n",temprd[y][x],tempr_lapd[y][x],tempr_xs1,tempr_xs2,tempr_ys1,tempr_ys2);
            //             printf("gram phi:%10e,phi_lap:%10e,xs1:%10e,xs2:%10e,ys1:%10e,ys2:%10e\n",phid[y][x],phi_lapd[y][x],phi_xs1,phi_xs2,phi_ys1,phi_ys2);
            //         }
            //     #endif
            // #endif
        }
    }
    __global__ void mix_kernel2(highprecision* phi,highprecision* phi_lap,highprecision* epsilon,highprecision *epsilon_deri,highprecision* phidx,highprecision* phidy,highprecision* tempr,highprecision* tempr_lap,lowprecision* hphi,lowprecision* hphi_lap,lowprecision* hepsilon,lowprecision *hepsilon_deri,lowprecision* hphidx,lowprecision* hphidy,lowprecision* htempr,lowprecision* htempr_lap,lowprecision hdym2,lowprecision hdxm2,lowprecision hgama,lowprecision hteq,lowprecision halpha,lowprecision hpi,lowprecision hone,lowprecision hzpf,lowprecision hdtime,lowprecision htau,lowprecision hkappa,int* type_curr,int i){
        int unitindex_x=blockIdx.z%unitdimX;
        int unitindex_y=blockIdx.z/unitdimX;
        int type;
        #ifdef AMSTENCIL
            int(*type_currd)[unitdimX]=(int(*)[unitdimX])type_curr;
            type=type_currd[unitindex_y][unitindex_x];
            if(type==1&&blockIdx.x==1)return;
        #endif
        #ifdef GRAM1
            if(blockIdx.z<unitNums*ratio)type=1;
            else type=2;
        #endif
        #ifdef GRAM2
            if(blockIdx.z%100<100*ratio)type=1;
            else type=2;
        #endif
        int unitindex_ys1=unitindex_y==0?unitdimY-1:unitindex_y-1;
        int unitindex_ya1=unitindex_y==unitdimY-1?0:unitindex_y+1;
        int x_offset=threadIdx.x;
        int x_start=unitindex_x*uxd2;
        int y_offset=threadIdx.y;
        int y=unitindex_y*unity+y_offset;
        int ys1=y>0?y-1:dimY-1;
        int ya1=y<dimY-1?y+1:0;
        lowprecision(*hphid)[lowprecison_dimX]=(lowprecision(*)[lowprecison_dimX])hphi;
        lowprecision(*htemprd)[lowprecison_dimX]=(lowprecision(*)[lowprecison_dimX])htempr;
        lowprecision(*hphi_lapd)[lowprecison_dimX]=(lowprecision(*)[lowprecison_dimX])hphi_lap;
        lowprecision(*hepsilond)[lowprecison_dimX]=(lowprecision(*)[lowprecison_dimX])hepsilon;
        lowprecision(*hepsilon_derid)[lowprecison_dimX]=(lowprecision(*)[lowprecison_dimX])hepsilon_deri;
        lowprecision(*hphidxd)[lowprecison_dimX]=(lowprecision(*)[lowprecison_dimX])hphidx;
        lowprecision(*hphidyd)[lowprecison_dimX]=(lowprecision(*)[lowprecison_dimX])hphidy;
        lowprecision(*htempr_lapd)[lowprecison_dimX]=(lowprecision(*)[lowprecison_dimX])htempr_lap;
        highprecision(*temprd)[dimX]=(highprecision(*)[dimX])tempr;
        highprecision(*phid)[dimX]=(highprecision(*)[dimX])phi;
        highprecision(*epsilond)[dimX]=(highprecision(*)[dimX])epsilon;
        highprecision(*epsilon_derid)[dimX]=(highprecision(*)[dimX])epsilon_deri;
        highprecision(*phidxd)[dimX]=(highprecision(*)[dimX])phidx;
        highprecision(*tempr_lapd)[dimX]=(highprecision(*)[dimX])tempr_lap;
        highprecision(*phidyd)[dimX]=(highprecision(*)[dimX])phidy;
        highprecision(*phi_lapd)[dimX]=(highprecision(*)[dimX])phi_lap;
        if(type==1){
            int unitindex_xs1=unitindex_x==0?unitdimX-1:unitindex_x-1;
            int unitindex_xa1=unitindex_x==unitdimX-1?0:unitindex_x+1;
            #ifdef AMSTENCIL
                int x=x_start+x_offset;
            #else
                x_offset=x_offset+blockIdx.x*uxd2;
                int x=x_start*2+x_offset;
            #endif
            int xs1=x>0?x-1:lowprecison_dimX-1;
            int xa1=x<lowprecison_dimX-1?x+1:0;
            lowprecision phi_old=hphid[y][x];
            lowprecision phidxd_ya1,epsilon_derid_ya1,epsilond_ya1,phidxd_ys1,epsilon_derid_ys1,epsilond_ys1,epsilond_xs1,epsilon_derid_xs1,phidyd_xs1,epsilond_xa1,epsilon_derid_xa1,phidyd_xa1;
            #ifdef AMSTENCIL
                if(y_offset<uys1||type_currd[unitindex_ya1][unitindex_x]==1){
                    phidxd_ya1=hphidxd[ya1][x];
                    epsilond_ya1=hepsilond[ya1][x];
                    epsilon_derid_ya1=hepsilon_derid[ya1][x];
                }
                else{
                    phidxd_ya1=__floats2half2_rn(phidxd[ya1][x*2],phidxd[ya1][x*2+1]);
                    epsilond_ya1=__floats2half2_rn(epsilond[ya1][x*2],epsilond[ya1][x*2+1]);
                    epsilon_derid_ya1=__floats2half2_rn(epsilon_derid[ya1][x*2],epsilon_derid[ya1][x*2+1]);
                }

                if(y_offset>0||type_currd[unitindex_ys1][unitindex_x]==1){
                    phidxd_ys1=hphidxd[ys1][x];
                    epsilon_derid_ys1=hepsilon_derid[ys1][x];
                    epsilond_ys1=hepsilond[ys1][x];
                }
                else{
                    phidxd_ys1=__floats2half2_rn(phidxd[ys1][x*2],phidxd[ys1][x*2+1]);
                    epsilon_derid_ys1=__floats2half2_rn(epsilon_derid[ys1][x*2],epsilon_derid[ys1][x*2+1]);
                    epsilond_ys1=__floats2half2_rn(epsilond[ys1][x*2],epsilond[ys1][x*2+1]);
                }

                if(x_offset>0||type_currd[unitindex_y][unitindex_xs1]==1){
                    epsilond_xs1=__halves2half2(__high2half(hepsilond[y][xs1]),__low2half(hepsilond[y][x]));
                    epsilon_derid_xs1=__halves2half2(__high2half(hepsilon_derid[y][xs1]),__low2half(hepsilon_derid[y][x]));
                    phidyd_xs1=__halves2half2(__high2half(hphidyd[y][xs1]),__low2half(hphidyd[y][x]));
                }
                else{
                    epsilond_xs1=__halves2half2(__float2half(epsilond[y][2*xs1+1]),__low2half(hepsilond[y][x]));
                    epsilon_derid_xs1=__halves2half2(__float2half(epsilon_derid[y][2*xs1+1]),__low2half(hepsilon_derid[y][x]));
                    phidyd_xs1=__halves2half2(__float2half(phidyd[y][2*xs1+1]),__low2half(hphidyd[y][x]));
                }

                if(x_offset<uxd2s1||type_currd[unitindex_y][unitindex_xa1]==1){
                    epsilond_xa1=__halves2half2(__high2half(hepsilond[y][x]),__low2half(hepsilond[y][xa1]));
                    epsilon_derid_xa1=__halves2half2(__high2half(hepsilon_derid[y][x]),__low2half(hepsilon_derid[y][xa1]));
                    phidyd_xa1=__halves2half2(__high2half(hphidyd[y][x]),__low2half(hphidyd[y][xa1]));
                }
                else{
                    epsilond_xa1=__halves2half2(__high2half(hepsilond[y][x]),__float2half(epsilond[y][2*xa1]));
                    epsilon_derid_xa1=__halves2half2(__high2half(hepsilon_derid[y][x]),__float2half(epsilon_derid[y][2*xa1]));
                    phidyd_xa1=__halves2half2(__high2half(hphidyd[y][x]),__float2half(phidyd[y][2*xa1]));
                }
            #endif
            #ifdef GRAM1
                epsilond_xs1=(x_offset>0||(unitindex_y*unitdimX+unitindex_xs1)<unitNums*ratio)?hepsilond[y][xs1]:__float2half(epsilond[y][xs1]);
                epsilond_xa1=(x_offset<uxs1||(unitindex_y*unitdimX+unitindex_xa1)<unitNums*ratio)?hepsilond[y][xa1]:__float2half(epsilond[y][xa1]);
                epsilond_ys1=(y_offset>0||(unitindex_ys1*unitdimX+unitindex_x)<unitNums*ratio)?hepsilond[ys1][x]:__float2half(epsilond[ys1][x]);
                epsilond_ya1=(y_offset<uys1||(unitindex_ya1*unitdimX+unitindex_x)<unitNums*ratio)?hepsilond[ya1][x]:__float2half(epsilond[ya1][x]);
                epsilon_derid_xs1=(x_offset>0||(unitindex_y*unitdimX+unitindex_xs1)<unitNums*ratio)?hepsilon_derid[y][xs1]:__float2half(epsilon_derid[y][xs1]);
                epsilon_derid_xa1=(x_offset<uxs1||(unitindex_y*unitdimX+unitindex_xa1)<unitNums*ratio)?hepsilon_derid[y][xa1]:__float2half(epsilon_derid[y][xa1]);
                epsilon_derid_ys1=(y_offset>0||(unitindex_ys1*unitdimX+unitindex_x)<unitNums*ratio)?hepsilon_derid[ys1][x]:__float2half(epsilon_derid[ys1][x]);
                epsilon_derid_ya1=(y_offset<uys1||(unitindex_ya1*unitdimX+unitindex_x)<unitNums*ratio)?hepsilon_derid[ya1][x]:__float2half(epsilon_derid[ya1][x]);
                phidyd_xs1=(x_offset>0||(unitindex_y*unitdimX+unitindex_xs1)<unitNums*ratio)?hphidyd[y][xs1]:__float2half(phidyd[y][xs1]);
                phidyd_xa1=(x_offset<uxs1||(unitindex_y*unitdimX+unitindex_xa1)<unitNums*ratio)?hphidyd[y][xa1]:__float2half(phidyd[y][xa1]);
                phidxd_ys1=(y_offset>0||(unitindex_ys1*unitdimX+unitindex_x)<unitNums*ratio)?hphidxd[ys1][x]:__float2half(phidxd[ys1][x]);
                phidxd_ya1=(y_offset<uys1||(unitindex_ya1*unitdimX+unitindex_x)<unitNums*ratio)?hphidxd[ya1][x]:__float2half(phidxd[ya1][x]);
            #endif
            #ifdef GRAM2
                epsilond_xs1=(x_offset>0||(unitindex_y*unitdimX+unitindex_xs1)%100<100*ratio)?hepsilond[y][xs1]:__float2half(epsilond[y][xs1]);
                epsilond_xa1=(x_offset<uxs1||(unitindex_y*unitdimX+unitindex_xa1)%100<100*ratio)?hepsilond[y][xa1]:__float2half(epsilond[y][xa1]);
                epsilond_ys1=(y_offset>0||(unitindex_ys1*unitdimX+unitindex_x)%100<100*ratio)?hepsilond[ys1][x]:__float2half(epsilond[ys1][x]);
                epsilond_ya1=(y_offset<uys1||(unitindex_ya1*unitdimX+unitindex_x)%100<100*ratio)?hepsilond[ya1][x]:__float2half(epsilond[ya1][x]);
                epsilon_derid_xs1=(x_offset>0||(unitindex_y*unitdimX+unitindex_xs1)%100<100*ratio)?hepsilon_derid[y][xs1]:__float2half(epsilon_derid[y][xs1]);
                epsilon_derid_xa1=(x_offset<uxs1||(unitindex_y*unitdimX+unitindex_xa1)%100<100*ratio)?hepsilon_derid[y][xa1]:__float2half(epsilon_derid[y][xa1]);  
                epsilon_derid_ys1=(y_offset>0||(unitindex_ys1*unitdimX+unitindex_x)%100<100*ratio)?hepsilon_derid[ys1][x]:__float2half(epsilon_derid[ys1][x]);
                epsilon_derid_ya1=(y_offset<uys1||(unitindex_ya1*unitdimX+unitindex_x)%100<100*ratio)?hepsilon_derid[ya1][x]:__float2half(epsilon_derid[ya1][x]);  
                phidyd_xs1=(x_offset>0||(unitindex_y*unitdimX+unitindex_xs1)%100<100*ratio)?hphidyd[y][xs1]:__float2half(phidyd[y][xs1]);
                phidyd_xa1=(x_offset<uxs1||(unitindex_y*unitdimX+unitindex_xa1)%100<100*ratio)?hphidyd[y][xa1]:__float2half(phidyd[y][xa1]);
                phidxd_ys1=(y_offset>0||(unitindex_ys1*unitdimX+unitindex_x)%100<100*ratio)?hphidxd[ys1][x]:__float2half(phidxd[ys1][x]);
                phidxd_ya1=(y_offset<uys1||(unitindex_ya1*unitdimX+unitindex_x)%100<100*ratio)?hphidxd[ya1][x]:__float2half(phidxd[ya1][x]);  
            #endif
            lowprecision term1=(epsilond_ya1*epsilon_derid_ya1*phidxd_ya1-epsilond_ys1*epsilon_derid_ys1*phidxd_ys1)/hdym2;
            lowprecision term2=-(epsilond_xa1*epsilon_derid_xa1*phidyd_xa1-epsilond_xs1*epsilon_derid_xs1*phidyd_xs1)/hdxm2;
            #ifdef AMSTENCIL
                lowprecision tmp=hgama*(hteq-htemprd[y][x]);
                tmp=__floats2half2_rn(atan(__low2float(tmp)),atan(__high2float(tmp)));
            #else
                lowprecision tmp=__float2half(atan(__half2float(hgama*(hteq-htemprd[y][x]))));
            #endif
            lowprecision m=halpha/hpi*tmp;
            hphid[y][x]=hphid[y][x]+(hdtime/htau)*(term1+term2+(hepsilond[y][x]*hepsilond[y][x])*hphi_lapd[y][x]+phi_old*(hone-phi_old)*(phi_old-hzpf+m));
            htemprd[y][x]=htemprd[y][x]+hdtime*htempr_lapd[y][x]+hkappa*(hphid[y][x]-phi_old);
            #ifdef monitor_conversion_dependent
                #ifdef AMSTENCIL
                    if(y_offset<2||y_offset+2>uys1||x_offset==0||x_offset==uxd2s1||(i+1)%10==0){
                        phid[y][x*2]=__low2float(hphid[y][x]);
                        phid[y][x*2+1]=__high2float(hphid[y][x]);
                        temprd[y][x*2]=__low2float(htemprd[y][x]);
                        temprd[y][x*2+1]=__high2float(htemprd[y][x]);
                    }
                #endif
            #endif
            #ifdef PRINT_INFO
                #ifdef AMSTENCIL
                    if(x==321/2 and y==254 and i==110){
                        // printf("amst tempr%10e,tempr_lap:%10e,phi:%10e,phiold:%10e\n",__high2float(htemprd[y][x]),__high2float(htempr_lapd[y][x]),__high2float(hphid[y][x]),__high2float(phi_old));
                        // printf("amst phi:%10e,term1:%10e,term2:%10e,epsilon:%10e,philap:%10e,phiold:%10e,m:%10e\n",__high2float(hphid[y][x]),__high2float(term1),__high2float(term2),__high2float(hepsilond[y][x]),__high2float(hphi_lapd[y][x]),__high2float(phi_old),__high2float(m));
                    }
                #else
                    if(x==321 and y==254 and i==110){
                        // printf("gram tempr%10e,tempr_lap:%10e,phi:%10e,phiold:%10e\n",(float)htemprd[y][x],(float)htempr_lapd[y][x],(float)hphid[y][x],(float)phi_old);
                        // printf("gram phi:%10e,term1:%10e,term2:%10e,epsilon:%10e,philap:%10e,phiold:%10e,m:%10e\n",(float)hphid[y][x],(float)term1,(float)term2,(float)hepsilond[y][x],(float)hphi_lapd[y][x],(float)phi_old,(float)m);
                    }
                #endif
            #endif
        }
        else{
            
            x_offset=x_offset+blockIdx.x*uxd2;
            int x=x_start*2+x_offset;
            int xs1=x>0?x-1:dimX-1;
            int xa1=x<dimX-1?x+1:0;
            highprecision phi_old=phid[y][x];
            #ifdef AMSTENCIL
                highprecision term1=(epsilond[ya1][x]*epsilon_derid[ya1][x]*phidxd[ya1][x]-epsilond[ys1][x]*epsilon_derid[ys1][x]*phidxd[ys1][x])/(2.0*dy);
                highprecision term2=-(epsilond[y][xa1]*epsilon_derid[y][xa1]*phidyd[y][xa1]-epsilond[y][xs1]*epsilon_derid[y][xs1]*phidyd[y][xs1])/(2.0*dx);
            #endif
            #ifdef GRAM1
                int unitindex_xs1=unitindex_x==0?unitdimX-1:unitindex_x-1;
                int unitindex_xa1=unitindex_x==unitdimX-1?0:unitindex_x+1;
                highprecision epsilond_xs1=(x_offset>0||(unitindex_y*unitdimX+unitindex_xs1)>=unitNums*ratio)?epsilond[y][xs1]:__half2float(hepsilond[y][xs1]);
                highprecision epsilond_xa1=(x_offset<uxs1||(unitindex_y*unitdimX+unitindex_xa1)>=unitNums*ratio)?epsilond[y][xa1]:__half2float(hepsilond[y][xa1]);
                highprecision epsilond_ys1=(y_offset>0||(unitindex_ys1*unitdimX+unitindex_x)>=unitNums*ratio)?epsilond[ys1][x]:__half2float(hepsilond[ys1][x]);
                highprecision epsilond_ya1=(y_offset<uys1||(unitindex_ya1*unitdimX+unitindex_x)>=unitNums*ratio)?epsilond[ya1][x]:__half2float(hepsilond[ya1][x]);
                highprecision epsilon_derid_xs1=(x_offset>0||(unitindex_y*unitdimX+unitindex_xs1)>=unitNums*ratio)?epsilon_derid[y][xs1]:__half2float(hepsilon_derid[y][xs1]);
                highprecision epsilon_derid_xa1=(x_offset<uxs1||(unitindex_y*unitdimX+unitindex_xa1)>=unitNums*ratio)?epsilon_derid[y][xa1]:__half2float(hepsilon_derid[y][xa1]);
                highprecision epsilon_derid_ys1=(y_offset>0||(unitindex_ys1*unitdimX+unitindex_x)>=unitNums*ratio)?epsilon_derid[ys1][x]:__half2float(hepsilon_derid[ys1][x]);
                highprecision epsilon_derid_ya1=(y_offset<uys1||(unitindex_ya1*unitdimX+unitindex_x)>=unitNums*ratio)?epsilon_derid[ya1][x]:__half2float(hepsilon_derid[ya1][x]);
                highprecision phidyd_xs1=(x_offset>0||(unitindex_y*unitdimX+unitindex_xs1)>=unitNums*ratio)?phidyd[y][xs1]:__half2float(hphidyd[y][xs1]);
                highprecision phidyd_xa1=(x_offset<uxs1||(unitindex_y*unitdimX+unitindex_xa1)>=unitNums*ratio)?phidyd[y][xa1]:__half2float(hphidyd[y][xa1]);
                highprecision phidxd_ys1=(y_offset>0||(unitindex_ys1*unitdimX+unitindex_x)>=unitNums*ratio)?phidxd[ys1][x]:__half2float(hphidxd[ys1][x]);
                highprecision phidxd_ya1=(y_offset<uys1||(unitindex_ya1*unitdimX+unitindex_x)>=unitNums*ratio)?phidxd[ya1][x]:__half2float(hphidxd[ya1][x]);
                highprecision term1=(epsilond_ya1*epsilon_derid_ya1*phidxd_ya1-epsilond_ys1*epsilon_derid_ys1*phidxd_ys1)/(2.0*dy);
                highprecision term2=-(epsilond_xa1*epsilon_derid_xa1*phidyd_xa1-epsilond_xs1*epsilon_derid_xs1*phidyd_xs1)/(2.0*dx);
            #endif
            #ifdef GRAM2
                int unitindex_xs1=unitindex_x==0?unitdimX-1:unitindex_x-1;
                int unitindex_xa1=unitindex_x==unitdimX-1?0:unitindex_x+1;
                highprecision epsilond_xs1=(x_offset>0||(unitindex_y*unitdimX+unitindex_xs1)%100>=100*ratio)?epsilond[y][xs1]:__half2float(hepsilond[y][xs1]);
                highprecision epsilond_xa1=(x_offset<uxs1||(unitindex_y*unitdimX+unitindex_xa1)%100>=100*ratio)?epsilond[y][xa1]:__half2float(hepsilond[y][xa1]);
                highprecision epsilond_ys1=(y_offset>0||(unitindex_ys1*unitdimX+unitindex_x)%100>=100*ratio)?epsilond[ys1][x]:__half2float(hepsilond[ys1][x]);
                highprecision epsilond_ya1=(y_offset<uys1||(unitindex_ya1*unitdimX+unitindex_x)%100>=100*ratio)?epsilond[ya1][x]:__half2float(hepsilond[ya1][x]);
                highprecision epsilon_derid_xs1=(x_offset>0||(unitindex_y*unitdimX+unitindex_xs1)%100>=100*ratio)?epsilon_derid[y][xs1]:__half2float(hepsilon_derid[y][xs1]);
                highprecision epsilon_derid_xa1=(x_offset<uxs1||(unitindex_y*unitdimX+unitindex_xa1)%100>=100*ratio)?epsilon_derid[y][xa1]:__half2float(hepsilon_derid[y][xa1]);
                highprecision epsilon_derid_ys1=(y_offset>0||(unitindex_ys1*unitdimX+unitindex_x)%100>=100*ratio)?epsilon_derid[ys1][x]:__half2float(hepsilon_derid[ys1][x]);
                highprecision epsilon_derid_ya1=(y_offset<uys1||(unitindex_ya1*unitdimX+unitindex_x)%100>=100*ratio)?epsilon_derid[ya1][x]:__half2float(hepsilon_derid[ya1][x]);
                highprecision phidyd_xs1=(x_offset>0||(unitindex_y*unitdimX+unitindex_xs1)%100>=100*ratio)?phidyd[y][xs1]:__half2float(hphidyd[y][xs1]);
                highprecision phidyd_xa1=(x_offset<uxs1||(unitindex_y*unitdimX+unitindex_xa1)%100>=100*ratio)?phidyd[y][xa1]:__half2float(hphidyd[y][xa1]);
                highprecision phidxd_ys1=(y_offset>0||(unitindex_ys1*unitdimX+unitindex_x)%100>=100*ratio)?phidxd[ys1][x]:__half2float(hphidxd[ys1][x]);
                highprecision phidxd_ya1=(y_offset<uys1||(unitindex_ya1*unitdimX+unitindex_x)%100>=100*ratio)?phidxd[ya1][x]:__half2float(hphidxd[ya1][x]);
                highprecision term1=(epsilond_ya1*epsilon_derid_ya1*phidxd_ya1-epsilond_ys1*epsilon_derid_ys1*phidxd_ys1)/(2.0*dy);
                highprecision term2=-(epsilond_xa1*epsilon_derid_xa1*phidyd_xa1-epsilond_xs1*epsilon_derid_xs1*phidyd_xs1)/(2.0*dx);
            #endif
            highprecision m=alpha/pi*atan(gama*(teq-temprd[y][x]));
            phid[y][x]=phid[y][x]+(dtime/tau)*(term1+term2+pow(epsilond[y][x],2)*phi_lapd[y][x]+phi_old*(1.0-phi_old)*(phi_old-0.5+m));
            temprd[y][x]=temprd[y][x]+dtime*tempr_lapd[y][x]+kappa*(phid[y][x]-phi_old);
            #ifdef PRINT_INFO
                #ifdef AMSTENCIL
                    if(x==263 and y==256 and i==4){
                        // printf("amst tempr%10e,tempr_lap:%10e,phi:%10e,phiold:%10e\n",(temprd[y][x]),(tempr_lapd[y][x]),(phid[y][x]),(phi_old));
                        printf("amst phi:%10e,term1:%10e,term2:%10e,epsilon:%10e,philap:%10e,phiold:%10e,m:%10e\n",phid[y][x],term1,term2,epsilond[y][x],phi_lapd[y][x],phi_old,m);
                        // printf("amst term1:%10e,epsilond_ya1:%10e,epsilon_derid_ya1:%10e,phidxd_ya1:%10e,epsilond_ys1:%10e,epsilon_derid_ys1:%10e,phidxd_ys1:%10e\n",term1,epsilond[ya1][x],epsilon_derid[ya1][x],phidxd[ya1][x],epsilond[ys1][x],epsilon_derid[ys1][x],phidxd[ys1][x]);
                    }
                #else
                    if(x==263 and y==256 and i==4){
                        // printf("gram tempr%10e,tempr_lap:%10e,phi:%10e,phiold:%10e\n",temprd[y][x],tempr_lapd[y][x],phid[y][x],phi_old);
                        printf("gram phi:%10e,term1:%10e,term2:%10e,epsilon:%10e,philap:%10e,phiold:%10e,m:%10e\n",phid[y][x],term1,term2,epsilond[y][x],phi_lapd[y][x],phi_old,m);
                        // printf("gram term1:%10e,epsilond_ya1:%10e,epsilon_derid_ya1:%10e,phidxd_ya1:%10e,epsilond_ys1:%10e,epsilon_derid_ys1:%10e,phidxd_ys1:%10e\n",term1,epsilond_ya1,epsilon_derid_ya1,phidxd_ya1,epsilond_ys1,epsilon_derid_ys1,phidxd_ys1);
                    }
                #endif
            #endif
        }
    }
    #endif
    #if ((defined HALF)||(defined HALF2))
        __global__ void kernel1_purelow(purelowprecision *phi,purelowprecision* phi_lap,purelowprecision* tempr,purelowprecision* tempr_lap,purelowprecision* phidx,purelowprecision* phidy,purelowprecision* epsilon,purelowprecision* epsilon_deri,purelowprecision hdxdy,purelowprecision htheta0,purelowprecision haniso,purelowprecision hone,purelowprecision hdxm2,purelowprecision hdym2,purelowprecision hdelta,purelowprecision hepsilonb,purelowprecision height){
            int unitindex_x=blockIdx.z%unitdimX;
            int unitindex_y=blockIdx.z/unitdimX;
            int x_offset=threadIdx.x ;
            int x_start;
            #ifdef HALF
                x_start=unitindex_x*unitx;
            #else
                x_start=unitindex_x*uxd2;
            #endif
            int x=x_start+x_offset;
            int y_offset=threadIdx.y;
            int y=unitindex_y*unity+y_offset;
            int xs1=x>0?x-1:purelowprecision_dimX-1;int ys1=y>0?y-1:dimY-1;
            int xa1=x<purelowprecision_dimX-1?x+1:0;int ya1=y<dimY-1?y+1:0;
            int ys2=y-2<0?y-2+dimY:y-2;
            int ya2=y+2>dimY-1?y+2-dimY:y+2;
            purelowprecision(*phid)[purelowprecision_dimX]=(purelowprecision(*)[purelowprecision_dimX])phi;
            purelowprecision(*phi_lapd)[purelowprecision_dimX]=(purelowprecision(*)[purelowprecision_dimX])phi_lap;
            purelowprecision(*phidxd)[purelowprecision_dimX]=(purelowprecision(*)[purelowprecision_dimX])phidx;
            purelowprecision(*phidyd)[purelowprecision_dimX]=(purelowprecision(*)[purelowprecision_dimX])phidy;
            purelowprecision(*temprd)[purelowprecision_dimX]=(purelowprecision(*)[purelowprecision_dimX])tempr;
            purelowprecision(*tempr_lapd)[purelowprecision_dimX]=(purelowprecision(*)[purelowprecision_dimX])tempr_lap;
            purelowprecision(*epsilond)[purelowprecision_dimX]=(purelowprecision(*)[purelowprecision_dimX])epsilon;
            purelowprecision(*epsilon_derid)[purelowprecision_dimX]=(purelowprecision(*)[purelowprecision_dimX])epsilon_deri;
            #ifdef HALF
                int xs2=x-2<0?x-2+purelowprecision_dimX:x-2;
                int xa2=x+2>purelowprecision_dimX-1?x+2-purelowprecision_dimX:x+2;
                phi_lapd[y][x]=phid[y][xs1]/hdxdy+phid[y][xs2]/hdxdy+phid[y][xa1]/hdxdy+phid[y][xa2]/hdxdy+phid[ys1][x]/hdxdy+phid[ys2][x]/hdxdy+phid[ya1][x]/hdxdy+phid[ya2][x]/hdxdy-height*phid[y][x]/hdxdy;
                tempr_lapd[y][x]=temprd[y][xs1]/hdxdy+temprd[y][xs2]/hdxdy+temprd[y][xa1]/hdxdy+temprd[y][xa2]/hdxdy+temprd[ys1][x]/hdxdy+temprd[ys2][x]/hdxdy+temprd[ya1][x]/hdxdy+temprd[ya2][x]/hdxdy-height*temprd[y][x]/hdxdy;
                phidxd[y][x]=(phid[y][xa1]-phid[y][xs1])/hdxm2;
            #else
                phi_lapd[y][x]=(__halves2half2(__high2half(phid[y][xs1]),__low2half(phid[y][x]))/hdxdy+phid[y][xs1]/hdxdy+__halves2half2(__high2half(phid[y][x]),__low2half(phid[y][xa1]))/hdxdy+phid[y][xa1]/hdxdy+phid[ys1][x]/hdxdy+phid[ys2][x]/hdxdy+phid[ya1][x]/hdxdy+phid[ya2][x]/hdxdy-height*phid[y][x]/hdxdy);      
                tempr_lapd[y][x]=(__halves2half2(__high2half(temprd[y][xs1]),__low2half(temprd[y][x]))/hdxdy+temprd[y][xs1]/hdxdy+__halves2half2(__high2half(temprd[y][x]),__low2half(temprd[y][xa1]))/hdxdy+temprd[y][xa1]/hdxdy+temprd[ys1][x]/hdxdy+temprd[ys2][x]/hdxdy+temprd[ya1][x]/hdxdy+temprd[ya2][x]/hdxdy-height*temprd[y][x]/hdxdy);  
                phidxd[y][x]=(__halves2half2(__high2half(phid[y][x]),__low2half(phid[y][xa1]))-__halves2half2(__high2half(phid[y][xs1]),__low2half(phid[y][x])))/hdxm2;
    
            #endif
            phidyd[y][x]=(phid[ya1][x]-phid[ys1][x])/hdym2;
            #ifdef HALF
                purelowprecision theta=__float2half(atan2(__half2float(phidyd[y][x]),__half2float(phidxd[y][x])));
                purelowprecision tmp=haniso*(theta-htheta0);
                purelowprecision tmp1=hcos(tmp);
                purelowprecision tmp2=hsin(tmp);
            #else
                purelowprecision theta=__floats2half2_rn(atan2(__low2float(phidyd[y][x]),__low2float(phidxd[y][x])),atan2(__high2float(phidyd[y][x]),__high2float(phidxd[y][x])));
                purelowprecision tmp=haniso*(theta-htheta0);
                purelowprecision tmp1=h2cos(tmp);
                purelowprecision tmp2=h2sin(tmp);
            #endif
            epsilond[y][x]=hepsilonb*(hone+hdelta*tmp1);
            epsilon_derid[y][x]=-hepsilonb*haniso*hdelta*tmp2;

        }
        __global__ void kernel2_purelow(purelowprecision* phi,purelowprecision* phi_lap,purelowprecision* epsilon,purelowprecision *epsilon_deri,purelowprecision* phidx,purelowprecision* phidy,purelowprecision* tempr,purelowprecision* tempr_lap,purelowprecision hdym2,purelowprecision hdxm2,purelowprecision hgama,purelowprecision hteq,purelowprecision halpha,purelowprecision hpi,purelowprecision hone,purelowprecision hzpf,purelowprecision hdtime,purelowprecision htau,purelowprecision hkappa
        ){
            int unitindex_x=blockIdx.z%unitdimX;
            int unitindex_y=blockIdx.z/unitdimX;
            int x_offset=threadIdx.x ;
            int x_start;
            #ifdef HALF
                x_start=unitindex_x*unitx;
            #else
                x_start=unitindex_x*uxd2;
            #endif
            int x=x_start+x_offset;
            int y_offset=threadIdx.y;
            int y=unitindex_y*unity+y_offset;
            int xs1=x>0?x-1:purelowprecision_dimX-1;int ys1=y>0?y-1:dimY-1;
            int xa1=x<purelowprecision_dimX-1?x+1:0;int ya1=y<dimY-1?y+1:0;
            purelowprecision(*phid)[purelowprecision_dimX]=(purelowprecision(*)[purelowprecision_dimX])phi;
            purelowprecision(*phi_lapd)[purelowprecision_dimX]=(purelowprecision(*)[purelowprecision_dimX])phi_lap;
            purelowprecision(*epsilond)[purelowprecision_dimX]=(purelowprecision(*)[purelowprecision_dimX])epsilon;
            purelowprecision(*epsilon_derid)[purelowprecision_dimX]=(purelowprecision(*)[purelowprecision_dimX])epsilon_deri;
            purelowprecision(*phidxd)[purelowprecision_dimX]=(purelowprecision(*)[purelowprecision_dimX])phidx;
            purelowprecision(*phidyd)[purelowprecision_dimX]=(purelowprecision(*)[purelowprecision_dimX])phidy;
            purelowprecision(*temprd)[purelowprecision_dimX]=(purelowprecision(*)[purelowprecision_dimX])tempr;
            purelowprecision(*tempr_lapd)[purelowprecision_dimX]=(purelowprecision(*)[purelowprecision_dimX])tempr_lap;

            purelowprecision phi_old=phid[y][x];
            purelowprecision term1=(epsilond[ya1][x]*epsilon_derid[ya1][x]*phidxd[ya1][x]-epsilond[ys1][x]*epsilon_derid[ys1][x]*phidxd[ys1][x])/hdym2;

            
            #ifdef HALF
                purelowprecision term2=-(epsilond[y][xa1]*epsilon_derid[y][xa1]*phidyd[y][xa1]-epsilond[y][xs1]*epsilon_derid[y][xs1]*phidyd[y][xs1])/hdxm2;
                purelowprecision tmp=__float2half(atan(__half2float(hgama*(hteq-temprd[y][x]))));
            #else 
                purelowprecision term2=-(__halves2half2(__high2half(epsilond[y][x]),__low2half(epsilond[y][xa1]))*__halves2half2(__high2half(epsilon_derid[y][x]),__low2half(epsilon_derid[y][xa1]))*__halves2half2(__high2half(phidyd[y][x]),__low2half(phidyd[y][xa1]))-__halves2half2(__high2half(epsilond[y][xs1]),__low2half(epsilond[y][x]))*__halves2half2(__high2half(epsilon_derid[y][xs1]),__low2half(epsilon_derid[y][x]))*__halves2half2(__high2half(phidyd[y][xs1]),__low2half(phidyd[y][x])))/hdxm2;
                purelowprecision tmp=hgama*(hteq-temprd[y][x]);
                tmp=__floats2half2_rn(atan(__low2float(tmp)),atan(__high2float(tmp)));
            #endif
            purelowprecision m=halpha/hpi*tmp;
            phid[y][x]=phid[y][x]+(hdtime/htau)*(term1+term2+(epsilond[y][x]*epsilond[y][x])*phi_lapd[y][x]+phi_old*(hone-phi_old)*(phi_old-hzpf+m));
            temprd[y][x]=temprd[y][x]+hdtime*tempr_lapd[y][x]+hkappa*(phid[y][x]-phi_old);
        }
    #endif

    #ifdef monitor_conversion_independent
        __global__ void monitor2_lastdata_store(highprecision* phi,highprecision *phi_last){
            int unitindex_x=blockIdx.z%unitdimX;
            int unitindex_y=blockIdx.z/unitdimX;
            int x_offset=threadIdx.x;
            int x_start=unitindex_x*uxd2;
            int y_offset=threadIdx.y;
            int y=unitindex_y*unity+y_offset;
            highprecision(*phid)[dimX]=(highprecision(*)[dimX])phi;
            highprecision(*phi_lastd)[dimX]=(highprecision(*)[dimX])phi_last;
            phi_lastd[y][x_start*2+x_offset+uxd2*blockIdx.x]=phid[y][x_start*2+x_offset+uxd2*blockIdx.x];
        }
        __global__ void kernel1_conversion(highprecision* epsilon,highprecision *epsilon_deri,highprecision* phidx,highprecision* phidy,lowprecision* hphidx,lowprecision* hphidy,lowprecision* hepsilon,lowprecision* hepsilon_deri,int* type_curr,int i){
            int unitindex_x=blockIdx.z%unitdimX;
            int unitindex_y=blockIdx.z/unitdimX;
            int type;
            int(*type_currd)[unitdimX]=(int(*)[unitdimX])type_curr;
            type=type_currd[unitindex_y][unitindex_x];
            if(type==1&&blockIdx.x==1)return;
            int x_offset=threadIdx.x;
            int x_start=unitindex_x*uxd2;
            int y_offset=threadIdx.y;
            int y=unitindex_y*unity+y_offset;
            lowprecision(*hphidxd)[lowprecison_dimX]=(lowprecision(*)[lowprecison_dimX])hphidx;
            lowprecision(*hphidyd)[lowprecison_dimX]=(lowprecision(*)[lowprecison_dimX])hphidy;
            lowprecision(*hepsilond)[lowprecison_dimX]=(lowprecision(*)[lowprecison_dimX])hepsilon;
            lowprecision(*hepsilon_derid)[lowprecison_dimX]=(lowprecision(*)[lowprecison_dimX])hepsilon_deri;
            highprecision(*phidxd)[dimX]=(highprecision(*)[dimX])phidx;
            highprecision(*phidyd)[dimX]=(highprecision(*)[dimX])phidy;
            highprecision(*epsilond)[dimX]=(highprecision(*)[dimX])epsilon;
            highprecision(*epsilon_derid)[dimX]=(highprecision(*)[dimX])epsilon_deri;
            if(type==1){
                int x=x_start+x_offset;
                if(y_offset==0||y_offset==uys1){
                    phidxd[y][x*2]=__low2float(hphidxd[y][x]);
                    phidxd[y][x*2+1]=__high2float(hphidxd[y][x]);
                    phidyd[y][x*2]=__low2float(hphidyd[y][x]);
                    phidyd[y][x*2+1]=__high2float(hphidyd[y][x]);
                    epsilond[y][x*2]=__low2float(hepsilond[y][x]);
                    epsilond[y][x*2+1]=__high2float(hepsilond[y][x]);
                    epsilon_derid[y][x*2]=__low2float(hepsilon_derid[y][x]);
                    epsilon_derid[y][x*2+1]=__high2float(hepsilon_derid[y][x]);
                }
                else if(x_offset==0){
                    phidxd[y][x*2]=__low2float(hphidxd[y][x]);
                    phidyd[y][x*2]=__low2float(hphidyd[y][x]);
                    epsilond[y][x*2]=__low2float(hepsilond[y][x]);
                    epsilon_derid[y][x*2]=__low2float(hepsilon_derid[y][x]);
                }
                else if(x_offset==uxd2s1){
                    phidxd[y][x*2+1]=__high2float(hphidxd[y][x]);
                    phidyd[y][x*2+1]=__high2float(hphidyd[y][x]);
                    epsilond[y][x*2+1]=__high2float(hepsilond[y][x]);
                    epsilon_derid[y][x*2+1]=__high2float(hepsilon_derid[y][x]);
                }
            }
        }
        __global__ void kernel2_conversion(highprecision* phi,highprecision* tempr,lowprecision* hphi,lowprecision* htempr,int* type_curr,int i){
            int unitindex_x=blockIdx.z%unitdimX;
            int unitindex_y=blockIdx.z/unitdimX;
            int type;
            int(*type_currd)[unitdimX]=(int(*)[unitdimX])type_curr;
            type=type_currd[unitindex_y][unitindex_x];
            if(type==1&&blockIdx.x==1)return;
            int x_offset=threadIdx.x;
            int x_start=unitindex_x*uxd2;
            int y_offset=threadIdx.y;
            int y=unitindex_y*unity+y_offset;
            lowprecision(*hphid)[lowprecison_dimX]=(lowprecision(*)[lowprecison_dimX])hphi;
            lowprecision(*htemprd)[lowprecison_dimX]=(lowprecision(*)[lowprecison_dimX])htempr;
            highprecision(*temprd)[dimX]=(highprecision(*)[dimX])tempr;
            highprecision(*phid)[dimX]=(highprecision(*)[dimX])phi;
            if(type==1){
                int x=x_start+x_offset;
                if(y_offset<2||y_offset+2>uys1||x_offset==0||x_offset==uxd2s1||(i+1)%10==0){
                    phid[y][x*2]=__low2float(hphid[y][x]);
                    phid[y][x*2+1]=__high2float(hphid[y][x]);
                    temprd[y][x*2]=__low2float(htemprd[y][x]);
                    temprd[y][x*2+1]=__high2float(htemprd[y][x]);
                }
            }
        }
    #endif
#endif
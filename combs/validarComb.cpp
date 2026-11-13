#include <thread>
#include <stdio.h>
#include <vector>
#include <mutex>
#include <stdlib.h>
#include <string>
#include <omp.h>

using namespace std;

typedef struct comb
{
    int ids[5];
} COMB;

vector<COMB> base=vector<COMB>();
vector<COMB> combs=vector<COMB>();
vector<COMB> invalid=vector<COMB>();

mutex addLock;
mutex invLock;

bool validate(COMB c)
{
    for (COMB reg : combs)
    {
        int i=0;
        for (int j = 0; j < 5; j++)
        {
            //printf("%d ",c.ids[j]);
            for (int k = 0; k < 5; k++)
            {
                if (c.ids[j] == reg.ids[k])
                    i++;
                if(i>2)
                    return false;
            }
        }
        //printf("\n");
    }
    return true;
}

void check(int ini, int fin)
{
    printf("Hebra %d, inicio: %d, fin: %d\n",this_thread::get_id(), ini,fin);
    int k=ini;
    for(int i=ini;i<fin;i++){
        if(validate(base.at(i))){
            //addLock.lock();
            combs.push_back(base.at(i));
            //addLock.unlock();
        }else{
            //invLock.lock();
            invalid.push_back(base.at(i));
            //invLock.unlock();
        }
        if(k-i>1000){
            printf("Hebra %d +1000 revisiones\n",this_thread::get_id());
            k+=1000;
        }
    }
    printf("*Hebra %d, finalizado\n", this_thread::get_id());
}

int main()
{
    const char *path = "C:\\Users\\wings\\Documents\\NetBeansProjects\\Combinatorias\\Combinatoria.txt";
    FILE *arch = fopen(path, "r");
    int nThreads = 10;
    int i,cant=3000000, k=0;
    while(!feof(arch)&&k<cant*2){
        COMB c;
        char s[15];
        fscanf(arch,"%s\n",s);
        //printf("%s ",s);
        for(i=0;i<5;i++){
            int val=atoi(string(s).substr(i*3,3).c_str());
            c.ids[i]=val;
            //printf("%d ",c.ids[i]);
        }
        k++;
        //printf("\n");
        base.push_back(c);
    }
    printf("%d Combinaciones registradas.\n",base.size());
    fclose(arch);
    /*
    #pragma omp parallel num_threads(nThreads)
    {
    #pragma omp for private(i) schedule(runtime)    
    for(i=0;i<base.size();i++){
        printf("%d\n",i);
        if(validate(base.at(i))){
            combs.push_back(base.at(i));
        }else{
            invalid.push_back(base.at(i));
        }
    }
    }
    */

    vector <thread> hilos=vector<thread>();
    for(int i=0;i<nThreads;i++ ){
        hilos.emplace_back( thread(check,(int)i*(base.size()/nThreads),(int)(i+1)*(base.size()/nThreads)));
    }
    for(auto &h:hilos){
        if(h.joinable())
            h.join();
    }

    printf("Escribiendo combinatorias invalidas\n");
    arch=fopen("invalidas.txt","w");
    for(COMB in:invalid){
        for( i=0;i<5;i++){
            if(in.ids[i]<10){
                fprintf(arch,"00%d",in.ids[i]);
                //printf("00%d ",in.ids[i]);
            continue;    
            }
            if(in.ids[i]<100){
                fprintf(arch,"0%d",in.ids[i]);
                //printf("0%d ",in.ids[i]);    
            }
            if(in.ids[i]>=100){
                fprintf(arch,"%d",in.ids[i]);
                //printf("%d ",in.ids[i]); 
            }
        }
        //printf("\n");
        fprintf(arch,"\n");
    }
    fclose(arch);
    
    printf("Escribiendo combinatorias validas\n");
    arch=fopen("validas.txt","w");
    for(COMB in:combs){
        for(int i=0;i<5;i++){
            if(in.ids[i]<10){
                fprintf(arch,"00%d",in.ids[i]);
                //printf("00%d ",in.ids[i]);
            continue;    
            }
            if(in.ids[i]<100){
                fprintf(arch,"0%d",in.ids[i]);
                //printf("0%d ",in.ids[i]);    
            }
            if(in.ids[i]>=100){
                fprintf(arch,"%d",in.ids[i]);
                //printf("%d ",in.ids[i]); 
            }
        }
        //printf("\n");
        fprintf(arch,"\n");
    }
    fclose(arch);
}
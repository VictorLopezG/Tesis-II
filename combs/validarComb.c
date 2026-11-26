#include <stdio.h>
#include <stdlib.h>

typedef struct comb{
    int ids[5];
} COMB;


int validar(COMB* regs,COMB c, int n_reg){
    //printf("%d\n",n_reg);
    for(int i=0;i<n_reg;i++){
        int iguales=0;
        for(int j=0;j<5;j++){
            for(int k=0;k<5;k++){
                if(c.ids[j]==regs[i].ids[k]){
                    iguales++;
                    break;
                }
            }
            if(iguales>2)
                return 0;
        }
    }
    return 1;
}

void registrar(COMB* regs, COMB c, int *n_reg){
    int val=validar(regs,c,*n_reg);
    if(val){
        regs[*n_reg]=c;
        printf("%d ",*n_reg);
        *n_reg=*n_reg+1;
        printf("%d\n",*n_reg);
    }
}


int main()
{
    COMB *base=NULL;
    COMB *regs=NULL;
    int n_reg=0;
    char *path = "C:\\Users\\wings\\Documents\\NetBeansProjects\\Combinatorias\\Combinatoria.txt";
    FILE *arch = fopen(path, "r");
    
    if(arch==NULL){
        printf("Error al abrir el archivo\n");
        return 1;
    }

    int cant=4000000, k=0;
    base=(COMB*)malloc(sizeof(COMB)*cant);
    regs=(COMB*)malloc(sizeof(COMB)*cant);
    
    if(base==NULL||regs==NULL){
        printf("Error de memoria\n");
        return 1;
    }
    
    while(!feof(arch)&&k<cant){
        COMB c;
        char s[15];
        fscanf(arch,"%s\n",s);
        //printf("%s\n",s);
        for(int i=0;i<5;i++){
            int j=i*3;
            char sub[]={s[j],s[j+1],s[j+2],'\0'};
            int val=atoi(sub);
            c.ids[i]=val;
            //printf("%c %c %c\n",s[j],s[j+1],s[j+2]);
        }
        base[k]=c;
        k++;
    }

    printf("%d Combinaciones leidas.\n",k);
    fclose(arch);
    
    FILE *validas=fopen("validas.txt","w");
    
    for(int i=0;i<k;i++){
        registrar(regs,base[i],&n_reg);
        if(i%1000==0){
            printf("%d combinaciones registradas\n",i);
            //printf("%d %d %d %d %d\n",base[i].ids[0],base[i].ids[1],base[i].ids[2],base[i].ids[3],base[i].ids[4]);
        }
        //printf("%d\n",n_reg);
        for(int j=0;j<5;j++){
            if(base[i].ids[j]<10){
                fprintf(arch,"00%d",base[i].ids[j]);
                continue;    
            }
            if(base[i].ids[j]<100){
                fprintf(arch,"0%d",base[i].ids[j]);
            }
            if(base[i].ids[j]>=100){
                fprintf(arch,"%d",base[i].ids[j]);
            }
        }
        fprintf(validas,"\n");
        //printf("%d %d %d %d %d\n",base[i].ids[0],base[i].ids[1],base[i].ids[2],base[i].ids[3],base[i].ids[4]);
    }
    fclose(validas);
    return 0;
}
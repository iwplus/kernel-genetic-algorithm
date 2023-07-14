from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from copy import deepcopy

######## Judul ##################

print('\n')
print('\n')
print('####         ####    #######         ###########      ###########    ####     ####     ####')
print(' ####       ####    #### ####        ####    ####     ####   ####    ####     ####     ####')
print('  ####     ####    ####   ####       ####   ####      ####  ####     #############     ####')
print('   ####   ####    #############      #########        ########       #############     ####')
print('    #### ####    ####       ####     ####   ####      ####           ####     ####     ####')
print('     #######    ####         ####    ####    ####     ####           ####     ####     ####')

print('\n')
print('\n')

judul = 'Welcome to Varphi version 1.1'
judul2 = '=================================='
subjudul1 = 'This program can be use to do a modeling using STATISTICAL DOWNSCALING NON-PARAMETRIC KERNEL which consists of three main parts: '
subjudul2 = '(1) Kmeans clustering,'
subjudul3 = '(2) CART, dan'
subjudul4 = '(3) Non-parametric kernel regression, where optimal bandwidth (which minimized the GCV) determined using GENETICS ALGORITHM.'


print(judul.center(80))
print(judul2.center(80))
print('\n')
print(subjudul1.center(40))
print(subjudul2)
print(subjudul3)
print(subjudul4)
print('\n')
input('Press ENTER to continue!')
print('\n')

from netCDF4 import Dataset

pros = 0

while pros <1:
    pros2 = 0
    while pros2 < 1:
        print('\n')
        
        pilih = input('Do you want to read the data from netcdf file? (y/n) ')
        print('\n')
        if pilih == 'y':
            pros = 1
            pros2 = 1
            print('\n')
            lokasi = input('Input the location of the netcdf file, e.g. D:\data\data.nc : ')
            print('\n')
            fh = Dataset(lokasi, mode='r')
            
            longi = fh.variables['lon'][:]
            lati = fh.variables['lat'][:]
            print('longitude = ', longi)
            print('\n')
            print('latitude = ', lati)
            print('\n')

            panjang_data = len(fh.variables['precip'][0][0][:])
            print('\n')
            n_lokasi = int(input('Input the number of location of your data : '))
            print('\n')

            datanc = np.zeros((panjang_data,n_lokasi))
            longi = np.array(longi)
            lati = np.array(lati)
            
            for i in range(n_lokasi):
                indeks_long = float(input('Input the LONGITUDE you want : '))
                indeks_lat = float(input('Input the LATITUDE you want : '))

            

                b1 = abs(longi-indeks_long)
                b2 = abs(lati-indeks_lat)
                indeksx = np.array(b1).argmin()
                indeksy = np.array(b2).argmin()
                datanc[:,i] = np.transpose(fh.variables['precip'][indeksx][indeksy][:])

            print('\n')
            simpan_data = input('Input the location for saving the data, Ex. D:\output\out.csv atau D:\output.csv: ')
            print('\n')
            datanc = pd.DataFrame(datanc)
            datanc.to_csv(simpan_data)
            print('\n')
            print('Netcdf data is ready in the CSV format. Please check in ', simpan_data)
            print('\n')
            fh.close()
        elif pilih == 'n':
            pros = 1
            pros2 = 1
        else:
            pros2 = 0
            print('\n')
            print(' This choice is not available right now..')
            input(' Press ENTER to choose the other options...')
            print('\n')
henti = 0
while henti < 1:
    
    ###### output directori ##########
    print('\n')
    direktori = input("Input the directory where you want to save the result, Ex. D:\output\ ")
    print('\n')

    ########import data##############
    sumber = input('Input the data data, Ex. D:\data\data.csv : ')
    data = pd.read_csv(sumber)
    data = np.array(data)

    print('\n')

    X = np.array(data[0:len(data),3]).reshape(len(data),1) ### change it if necessary
    print('data for clustering = ',X)


    ######## Determine the number of clusters using Elbow method #############
    print('\n')
    input('Press ENTER to proceed....')
    print('\n')
    distorsi = []
    K = int(input('Input the maximum number of clusters : '))
    for k in range(1,K):
        kmeanModel = KMeans(n_clusters=k).fit(X)
        kmeanModel.fit(X)
        distorsi.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_, 'euclidean'), axis=1))/X.shape[0])

    ### the elbow plot #####
    plt.plot(range(1,K), distorsi, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Distorsi')
    plt.title('The elbow plot, hint: choose optimal number of cluster from horizontal axis which corresponds to the elbow-like point in the plot')
    print('\n')
    print('load the plot....')
    print('\n')
    plt.show()

    ### Euclidean distance function ###
    def dist(a,b):
        return np.linalg.norm(a-b)

    #the number of optimal cluster
    k=int(input('Input the optimal number of clusters : '))

    #generate the initial center of clusters
    C = np.zeros((k,1),dtype=np.float32)

    for i in range(k):
        for j in range(1):
            C[i,j]=np.random.randint(0,np.max(X)-20, size=1)

    C_old=np.zeros(C.shape)

    #clusters label
    clusters=np.zeros(len(X))

    error = dist(C,C_old)

    #Find the correct centers
    while error != 0:
        distances=np.zeros(len(C))
        for i in range(len(X)):
            for j in range(len(C)):
                distances[j] = dist(X[i],C[j])
                
            clusters[i]=np.argmin(distances)
       
            C_old=deepcopy(C)
       
        for i in range(k):
            points = [X[j] for j in range(len(X)) if clusters[j] == i]
            C[i] = np.mean(points, axis=0)
        error = dist(C,C_old)
    print('\n')
    print('Centers = ',C)
    print('\n')

    #print('Klaster data 1 : ',clusters[1]+1)
    W = np.hstack((X,clusters.reshape(len(X),1)))
    klas = pd.DataFrame(W)
    lokasi_cetak = input('input file name for cluster output (e.g. cluster.csv) : ')
    klas.to_csv(direktori+lokasi_cetak)

    print('\n')
    print('Clustering is done, check the result in ', direktori+lokasi_cetak)
    print('\n')
    input('Press ENTER to proceed the CART.... ')
    print('\n')
   
    ############ Classification And Regression Trees (CART) ###############################

    from sklearn import tree
    import pydotplus

    #############Dataset########

    data = pd.read_csv(direktori+lokasi_cetak)
    data = np.array(data)
    data = np.array(data[0:len(data),2]).reshape(len(data),1)
    Y = data
    #print(data)
    data2 = pd.read_csv(sumber)
    data2 = np.array(data2)
    nkolom = int(input('Input number of variables : '))
    data2 =np.array(data2[0:len(data2),4:nkolom+4]).reshape(len(data2),26) #### change it when necessary
    #print(data2)

    data = np.hstack((data2,data))
    #print(data)
    #print(data)

    ##### Classification using Gini Index #############
    print('\n')
    X = np.array(data[0:,0:nkolom])
    print('X = ',X)
    print('Ukuran X = ', X.shape)
    print('Y = ',Y)
    print(Y.shape)
    print('\n')
    
    #### input the pruning parameter #####
    ncabang = int(input('Input the maximum number of branch : '))
    nlevel = int(input('Input the maximum depth : '))
    nsample = int(input('Input minimum number of samples : '))
    klasifikasi = tree.DecisionTreeClassifier(splitter = 'random', max_leaf_nodes = ncabang, min_samples_leaf = nsample, max_depth = nlevel)
    klasifikasi.fit(X,Y)
    print('\n')
    print('Classification data = ')
    print(klasifikasi)
    print('\n')
    
    ######### record the result ##############

    with open("klasifikasi_data.txt", "w") as f:
        f = tree.export_graphviz(klasifikasi, out_file = f)

    with open("klasifikasi_data.dot", "w") as f:
        f = tree.export_graphviz(klasifikasi, out_file = f)

    lokasi_pohon = input('Input file name for CART Tree image, (e.g. nama-file.png) : ')

    dot_file = tree.export_graphviz(klasifikasi, out_file=None, filled=True, rounded=True)
    graph = pydotplus.graph_from_dot_data(dot_file)
    graph.write_png(direktori+lokasi_pohon)
    print('\n')
    print('CART process is done, the CART Tree can be found in ', direktori+lokasi_pohon)
    print('\n')
    input('Press ENTER to proceed for determining optimum bandwidth... ')
    print('\n')

    ############# Determine Optimum Bandwidth using Genetics Algorithm #######################

    #generate initial population
    import math

    data = pd.read_csv(direktori+lokasi_cetak)
    data = np.array(data)
    respon = np.array(data[:,1])
    print('respon = ',respon)
    print('\n')

    data2 = pd.read_csv(sumber)
    data2= np.array(data2)
    nvar = int(input('Input the number of independent variables of your model : '))
    index = np.zeros((nvar),dtype = int)
    for i in range(nvar):
        print('indepent variable index ',i+1)
        index[i] = int(input())

    x = np.zeros((len(data2),nvar))
    for i in range(len(data2)):
        for j in range(nvar):
            x[i,j] = data2[i,index[j]+3]

    print('\n')
    print('x = ',x)
    print('Data shape = ',x.shape)
    print('\n')
    
    gen = nvar
    npop = int(input('Input the number of bandwidth to be generated : '))
    p_mutasi = float(input('Input mutation probability : '))
    par = len(x)
    batasbawah = math.pow(par,-1/(4+gen))-0.5
    batasatas = math.pow(par,-1/(4+gen))+0.5
    pop = np.random.uniform(low = batasbawah, high = batasatas, size=(npop,gen))
    print('\n')
    print('Initial Bandwidth = ', pop)
    print('\n')
    ngenerasi = int(input('Input the maximum number of iteration : '))
    print('\n')
    generasi = 0

    print('\n')
    print('Please wait, optimum bandwidth search in progress...')
    print('\n')
   
    while generasi < ngenerasi: 
       
        #fungsi kernel
        #import math

        def kernel(u):
            if abs(u)<=1:
                return 0.9375*math.pow(1-math.pow(u,2),2)
            else:
                return 0



        ############# GCV = FITNESS################################
            

        def gcv1(respon,x,h):
            yk = respon
            
            n = len(yk)
            m = len(h)
        
            u = np.zeros((m,n,n))
            for l in range(m):
                for i in range(n):
                    for j in range(n):
                        u[l,j,i] = (x[j,l]-x[i,l])/h[l]

                

            k = np.zeros((n,n))
            fk = np.zeros((n))

            for i in range(n):
                fk[i] = 0
                for j  in range(n):
                    hkali = 1
                    kernelkali = 1
                    for l in range(m):
                        hkali = hkali*h[l]
                        kernelkali = kernelkali*kernel(u[l,i,j])

                    k[i,j] = (1/(n*hkali))*kernelkali
                    fk[i] = fk[i]+k[i,j]


            H = np.zeros((n,n))
            for i in range(n):
                for j  in range(n):
                    hkali = 1
                    kernelkali = 1
                    for l in range(m):
                        hkali = hkali*h[l]
                        kernelkali = kernelkali*kernel(u[l,i,j])

                    H[i,j] = (1/fk[i])*(1/(n*hkali))*kernelkali

            

            mh0 = np.mat(H)*np.transpose(np.mat(yk))
            
            w = np.transpose(np.mat(yk))-mh0
            

            atas = np.transpose(w)*w
            
            bawah = math.pow((1-np.matrix.trace(np.matrix(H)))/n,2)

            return atas/bawah



        
    # fitness of all 
        nilaifit = np.zeros((npop)) 
    
        for i in range(npop):
            nilaifit[i] = gcv1(respon,x,pop[i,:]) 

        print('\n')
        print('Nilai GCV semua bandwidth = ',nilaifit)
        print('\n')
    
        urutanpop = np.argsort(nilaifit) #### sorted from the smallest GCV
        print('\n')
        print('Sorted index of best to worst bandwidth = ', urutanpop)
        print('\n')
   
        bobot = np.zeros((npop))
        for i in range(npop):
            bobot[urutanpop[i]] = (npop-i)/(npop)

        

    #rank selection
        S = sum(bobot)
        
        #print('S = ',S,' dan r = ',r)

        idx_seleksi = []
        while True:
            r = np.random.uniform(low = 0, high = S, size=(1,))
            kum_prob = 0
            for i in range(npop):
                kum_prob = kum_prob + bobot[i]
                if r <= kum_prob:
                    idx_seleksi.append(i)
                else:
                    break
                
            if len(idx_seleksi)>0:
                break
    
            

        selected = np.zeros((len(idx_seleksi),gen))
        for i in range(len(idx_seleksi)):
            selected[i,:]=pop[idx_seleksi[i],:]


    #### Crossover #########
    
        jum_silang = int(math.ceil((npop-len(selected))/2))
        hasil_persilangan=np.zeros((int(2*jum_silang),gen))
        for i in range(jum_silang):
            indeks1 = np.random.randint(0,len(selected)-1)
            indeks2 = np.random.randint(0,len(selected)-1)
        
        
        
            if gen == 1:
                inds = 0
            else :
                inds = np.random.randint(0,gen-1)
                if inds == gen-1:
                    anak1 = selected[ind1,:]
                    anak2 = selected[ind2,:]
                else :
        
                    anak1 = np.concatenate((selected[indeks1,:inds],selected[indeks2,(inds+1):(gen-1)])) ####cek lagi
        
                    anak2 = np.concatenate((selected[indeks2,:inds],selected[indeks1,(inds+1):(gen-1)])) ###cek lagi
        
            hasil_persilangan[i,] = anak1
            hasil_persilangan[jum_silang+i,] = anak2
        

        

    #### new population
        pop_baru = np.zeros((int(len(selected)+(2*jum_silang)),gen))

        for i in range(int(len(selected))):
            pop_baru[i,] = selected[i,]
        
        for i in range(jum_silang):
            pop_baru[i+int(len(selected)),]=hasil_persilangan[i,]
            pop_baru[i+int(len(selected))+jum_silang,]=hasil_persilangan[i+jum_silang,]

        print('\n')
        print('Bandwidth baru : ',pop_baru)
        print('\n')
    #### mutation
        

        for i in range(len(pop_baru)):
            r_mutasi = np.random.uniform(low = 0, high = 1,size=(1,))
            if r_mutasi <= p_mutasi:
                if gen == 1:
                    ind1 = 0
                    ind2 = 0
                else:
                    ind1 = np.random.randint(0,gen-1)
                    ind2 = np.random.randint(0,gen-1)

                temp = pop_baru[i,ind1]
                pop_baru[i,ind1] = pop_baru[i,ind2]
                pop_baru[i,ind2] = temp
        print('\n')
        print('Mutation result : ',pop_baru)
        print('\n')
        pop = pop_baru
        npop = len(pop) #####salah satu pengaturannya sudah di sini
        generasi = generasi+1

        if generasi == ngenerasi:
            band = pd.DataFrame(pop)
            print('\n')
            lokasi_bandwidth = input('Input bandwidth optimal file name (e.g. nama-file.csv) : ')
            print('\n')
            band.to_csv(direktori+lokasi_bandwidth)
            print('\n')
            print('Optimum Bandwidth found. You can find it in ', direktori+lokasi_bandwidth)
            print('\n')
            
        print('\n')
        print('Please wait, predicted data generated based on optimum bandwidth...')
        print('\n')

    ######## Plot prediction vs original data ##########

    
    band = pd.read_csv(direktori+lokasi_bandwidth)
    band = np.array(band)
    h = np.array(band[0,1:len(band[1,:])])

    

    yk = respon
    n = len(yk)
    m = len(h)

    u = np.zeros((m,n,n))
    for l in range(m):
        for i in range(n):
            for j in range(n):
                u[l,j,i] = (x[j,l]-x[i,l])/h[l]

        #print('u[',l+1,']= ',u[l,:,:])

    k = np.zeros((n,n))
    fk = np.zeros((n))

    for i in range(n):
        fk[i] = 0
        for j  in range(n):
            hkali = 1
            kernelkali = 1
            for l in range(m):
                hkali = hkali*h[l]
                kernelkali = kernelkali*kernel(u[l,i,j])

            k[i,j] = (1/(n*hkali))*kernelkali
            fk[i] = fk[i]+k[i,j]


    #print('fk = ',fk)

    H = np.zeros((n,n))
    for i in range(n):
        for j  in range(n):
            hkali = 1
            kernelkali = 1
            for l in range(m):
                hkali = hkali*h[l]
                kernelkali = kernelkali*kernel(u[l,i,j])

            H[i,j] = (1/fk[i])*(1/(n*hkali))*kernelkali

    
    mh0 = np.mat(H)*np.transpose(np.mat(yk))

    

    input('Pause.... <ENTER> to load the plot')

    print('\n')

    print('Load plot.....')

    print('\n')


    #### Plot data and Prediction ####

    plt.plot(yk, '-o', label='Data asli')
    plt.plot(mh0, '-.', label='Prediksi')
    plt.legend()
    plt.show()

    hasil = pd.DataFrame(mh0)
    print('\n')
    lokasi_hasil = input('Input file name for the prediction data (e.g. nama-file.csv) : ')
    print('\n')
    hasil.to_csv(direktori+lokasi_hasil)

    henti_sub = 0
    while henti_sub < 1:
        print('\n')
        ins = input('Do you wish to re-calculate the model? (y/n) ')
        print('\n')
        if ins == 'y':
            henti_sub = 1
            henti = 0
        elif ins == 'n':
            henti_sub = 1
            henti = 1
        else:
            print('\n')
            print('This option is not available for now.')
            input('Press <ENTER> to choose the other options.')
            print('\n')
            henti_sub = 0


print('\n')
print('Thank you')
print('\n')
input('Press ENTER to exit')

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

judul = 'Selamat Datang di Varphi versi 1.1'
judul2 = '=================================='
subjudul1 = 'Software ini digunakan untuk melakukan pemodelan dengan STATISTICAL DOWNSCALING NON-PARAMETRIK KERNEL yang terdiri dari TIGA komponen UTAMA: '
subjudul2 = '(1) Kmeans clustering,'
subjudul3 = '(2) CART, dan'
subjudul4 = '(3) Pemodelan non-parametrik kernel, dimana pencarian bandwidth optimal(yang meminimumkan GCV) memanfaatkan ALGORITMA GENETIKA.'


print(judul.center(80))
print(judul2.center(80))
print('\n')
print(subjudul1.center(40))
print(subjudul2)
print(subjudul3)
print(subjudul4)
print('\n')
input('Tekan ENTER untuk melanjutkan!')
print('\n')

from netCDF4 import Dataset

pros = 0

while pros <1:
    pros2 = 0
    while pros2 < 1:
        print('\n')
        
        pilih = input('Apakah Anda ingin mengambil data dari file netcdf terlebih dahulu? (y/n) ')
        print('\n')
        if pilih == 'y':
            pros = 1
            pros2 = 1
            print('\n')
            lokasi = input('Masukkan lokasi dan nama file data, Ex. D:\data\data.nc : ')
            print('\n')
            fh = Dataset(lokasi, mode='r')
            
            longi = fh.variables['lon'][:]
            lati = fh.variables['lat'][:]
            print('longitude = ', longi)
            print('\n')
            #print('ukuran logi = ', longi.shape)
            print('latitude = ', lati)
            print('\n')
            #print('ukuran lati = ', lati.shape)
            #print('panjang lati = ', len(lati))

            #precip = fh.variables['precip']

            panjang_data = len(fh.variables['precip'][0][0][:])
            print('\n')
            n_lokasi = int(input('Masukkan banyaknya lokasi yang akan diambil datanya : '))
            print('\n')

            datanc = np.zeros((panjang_data,n_lokasi))
            longi = np.array(longi)
            lati = np.array(lati)
            
            for i in range(n_lokasi):
                indeks_long = float(input('Masukkan LONGITUDE yang diinginkan : '))
                indeks_lat = float(input('Masukkan LATITUDE yang diinginkan : '))

            

                b1 = abs(longi-indeks_long)
                b2 = abs(lati-indeks_lat)
                indeksx = np.array(b1).argmin()
                indeksy = np.array(b2).argmin()
              #  print('indeks longitude = ', indeksx)
                #print('indeks latitude = ', indeksy)
                datanc[:,i] = np.transpose(fh.variables['precip'][indeksx][indeksy][:])
               #print('data di lokasi = ', np.transpose(fh.variables['precip'][indeksx][indeksy][:]))
                #print('Ukuran data = ', len(fh.variables['precip'][indeksx][indeksy][:]))

                #print('data yang terambil = ', data)
            print('\n')
            simpan_data = input('Masukkan lokasi file data disimpan, Ex. D:\output\out.csv atau D:\output.csv: ')
            print('\n')
            datanc = pd.DataFrame(datanc)
            datanc.to_csv(simpan_data)
            print('\n')
            print('Pengambilan data dari netcdf selesai. Silahkan cek hasilnya di ', simpan_data)
            print('\n')
            fh.close()
        elif pilih == 'n':
            pros = 1
            pros2 = 1
        else:
            pros2 = 0
            print('\n')
            print('Pilihan tidak tersedia..')
            input(' Tekan ENTER untuk memilih ulang...')
            print('\n')
henti = 0
while henti < 1:
    
    ###### direktori output ##########
    print('\n')
    direktori = input("Masukkan lokasi foder file output akan disimpan, Ex. D:\output\ ")
    print('\n')

    ########import data##############
    sumber = input('Masukkan file data Ex. D:\data\data.csv : ')
    data = pd.read_csv(sumber)
    data = np.array(data)

    print('\n')

    X = np.array(data[0:len(data),3]).reshape(len(data),1)
    print('data yang akan dikelompokkan = ',X)


    ########penentuan jumlah klaster#############
    print('\n')
    input('Tekan enter untuk memulai proses pengelompokan data....')
    print('\n')
    distorsi = []
    K = int(input('Input banyaknya klaster maksimum : '))
    for k in range(1,K):
        kmeanModel = KMeans(n_clusters=k).fit(X)
        kmeanModel.fit(X)
        distorsi.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_, 'euclidean'), axis=1))/X.shape[0])

    ###plot elbow#####
    plt.plot(range(1,K), distorsi, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Distorsi')
    plt.title('Metode Elbow yang menunjukkan jumlah cluster optimal')
    print('\n')
    print('load gambar....')
    print('\n')
    plt.show()

    #definisikan jarak Euclid
    def dist(a,b):
        return np.linalg.norm(a-b)

    #banyak cluster
    k=int(input('Input banyaknya cluster optimal : '))

    #generate center awal
    C = np.zeros((k,1),dtype=np.float32)

    for i in range(k):
        for j in range(1):
            C[i,j]=np.random.randint(0,np.max(X)-20, size=1)

    #print(C)

    #simpan nilai centroid saat diupdate
    C_old=np.zeros(C.shape)

    #label cluster
    clusters=np.zeros(len(X))

    #jarak antara centroid lama dan centroid baru
    error = dist(C,C_old)

    #cari centroid yang tepat
    while error != 0:
        distances=np.zeros(len(C))
        #masukkan data ke cluster terdekat
        for i in range(len(X)):
            for j in range(len(C)):
                distances[j] = dist(X[i],C[j])
                
            clusters[i]=np.argmin(distances)
        #simpan centroid lama
            C_old=deepcopy(C)
        #tentukan centroid baru
        for i in range(k):
            points = [X[j] for j in range(len(X)) if clusters[j] == i]
            C[i] = np.mean(points, axis=0)
        error = dist(C,C_old)
    print('\n')
    print('Pusat data = ',C)
    print('\n')

    #print('Klaster data 1 : ',clusters[1]+1)
    W = np.hstack((X,clusters.reshape(len(X),1)))
    klas = pd.DataFrame(W)
    lokasi_cetak = input('input nama file untuk menyimpan klaster (Contoh: klaster.csv) : ')
    klas.to_csv(direktori+lokasi_cetak)

    print('\n')
    print('Clustering sudah selesai, silahkan cek file klaster data di ', direktori+lokasi_cetak)
    print('\n')
    input('Tekan ENTER untuk lanjut ke CART.... ')
    print('\n')
    ############CART###############################

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
    nkolom = int(input('Input banyaknya variabel : '))
    data2 =np.array(data2[0:len(data2),4:nkolom+4]).reshape(len(data2),26)
    #print(data2)

    data = np.hstack((data2,data))
    #print(data)
    #print(data)

    #####klasifikasi#############
    print('\n')
    X = np.array(data[0:,0:nkolom])
    print('X = ',X)
    print('Ukuran X = ', X.shape)
    print('Y = ',Y)
    print(Y.shape)
    print('\n')
    ####input parameter untuk pruning pohon#####
    ncabang = int(input('Input banyaknya cabang maksimal : '))
    nlevel = int(input('Input kedalaman maksimal pohon : '))
    nsample = int(input('Input minimum sampel : '))
    klasifikasi = tree.DecisionTreeClassifier(splitter = 'random', max_leaf_nodes = ncabang, min_samples_leaf = nsample, max_depth = nlevel)
    klasifikasi.fit(X,Y)
    print('\n')
    print('Klasifikasi data = ')
    print(klasifikasi)
    print('\n')
    
    #########catat hasil##############

    with open("klasifikasi_data.txt", "w") as f:
        f = tree.export_graphviz(klasifikasi, out_file = f)

    #untuk cetak ke pdf
    with open("klasifikasi_data.dot", "w") as f:
        f = tree.export_graphviz(klasifikasi, out_file = f)

    lokasi_pohon = input('Input nama file untuk menyimpan pohon CART, (contoh: nama-file.png) : ')

    dot_file = tree.export_graphviz(klasifikasi, out_file=None, filled=True, rounded=True)
    graph = pydotplus.graph_from_dot_data(dot_file)
    graph.write_png(direktori+lokasi_pohon)
    print('\n')
    print('Proses CART selesai, gambar pohon dapat dilihat di ', direktori+lokasi_pohon)
    print('\n')
    input('Tekan ENTER untuk melanjutkan ke proses pencarian bandwidth... ')
    print('\n')

    ############# Pencarian Bandwidth dengan Algoritma Genetika #######################

    #algoritma genetika untuk mencari bandwidth optimum

    #generate populasi awal
    import math

    data = pd.read_csv(direktori+lokasi_cetak)
    data = np.array(data)
    respon = np.array(data[:,1])
    print('respon = ',respon)
    print('\n')

    data2 = pd.read_csv(sumber)
    data2= np.array(data2)
    nvar = int(input('Input banyaknya variabel yang masuk ke dalam model : '))
    index = np.zeros((nvar),dtype = int)
    for i in range(nvar):
        print('Input indeks variabel ke- ',i+1)
        index[i] = int(input())

    #print('Kumpulan indeks ', index)

    x = np.zeros((len(data2),nvar))
    for i in range(len(data2)):
        for j in range(nvar):
            x[i,j] = data2[i,index[j]+3]

    print('\n')
    print('x = ',x)
    print('Ukuran data = ',x.shape)
    print('\n')
    
    gen = nvar
    npop = int(input('Input banyaknya bandwidth yang akan dibangkitkan : '))
    p_mutasi = float(input('Masukkan peluang mutasi : '))
    par = len(x)
    batasbawah = math.pow(par,-1/(4+gen))-0.5
    batasatas = math.pow(par,-1/(4+gen))+0.5
    pop = np.random.uniform(low = batasbawah, high = batasatas, size=(npop,gen))
    print('\n')
    print('Bandwidth awal = ', pop)
    print('\n')
    ngenerasi = int(input('Input banyaknya iterasi yang diinginkan : '))
    print('\n')
    generasi = 0

    print('\n')
    print('Silahkan tunggu, pencarian bandwidth optimal sedang dalam proses...')
    print('\n')
    ### iterasi ######
    while generasi < ngenerasi:  ######atur lagi peletakan variabel npop dan nilaifit, karena harusnya berubah sesuai dengan jumlah individu
       
        #fungsi kernel
        #import math

        def kernel(u):
            if abs(u)<=1:
                return 0.9375*math.pow(1-math.pow(u,2),2)
            else:
                return 0



        #############fungsi GCV = fungsi FITNESS################################
            
        #import numpy as np

        
        #h = np.array([0.5, 0.2])

        def gcv1(respon,x,h):
            yk = respon
            #x1 = xa
            #x2 = xb
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

            #print('H = ',H)
            #print('trace H = ',np.matrix.trace(np.matrix(H)))

            mh0 = np.mat(H)*np.transpose(np.mat(yk))
            #print('mh0 = ',mh0)

            w = np.transpose(np.mat(yk))-mh0
            #print('yk-mh0 = ',w)

            atas = np.transpose(w)*w
            #print('atas = ',atas)

            bawah = math.pow((1-np.matrix.trace(np.matrix(H)))/n,2)
            #print('bawah = ',bawah)

            return atas/bawah



        #def fitness(h):

         #   fit = 0
          #  for i in range(len(h)):
           #     fit = fit + h[i]

            #return fit

    #indeks = int(input('Input indeks individu yang ingin dihitung fitnessnya : '))

    #h = pop[indeks,:]
    #fit = fitness(h)
    #print('Nilai fitness individu ke-',indeks,' adalah ',fit)

    #hitung fitness semua individu
        nilaifit = np.zeros((npop)) ###### nilaifit sudah sesuai
    #print(nilaifit)
        for i in range(npop):
            nilaifit[i] = gcv1(respon,x,pop[i,:]) ####fitness = GCV

        print('\n')
        print('Nilai GCV semua bandwidth = ',nilaifit)
        print('\n')
    #sort nilai fitness
        urutanpop = np.argsort(nilaifit) #### diurutkan dari GCV yang paling kecil
        print('\n')
        print('Urutan bandwidth dari terbaik sampai terburuk = ', urutanpop)
        print('\n')
    ##pemberian bobot berdasarkan rangking
        bobot = np.zeros((npop))
        for i in range(npop):
            bobot[urutanpop[i]] = (npop-i)/(npop)

        #print('Bobot masing-masing individu = ', bobot)

    #seleksi rangking
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
    ##
    ##    iterasi = np.math.floor(gen/2)
    ##    selected = np.zeros((iterasi,gen))
    ##    for i in range(iterasi):
    ##        r = np.random.uniform(low = 0, high = S, size=(1,))
    ##        kumulatif = 0
    ##        indeks = 0
    ##        for j in range(npop):
    ##            kumulatif = kumulatif + bobot[j]
    ##            if kumulatif<=r:
    ##                indeks = indeks + 1
    ##
    ##        selected[i,:] = pop[indeks,:]
        
            

        selected = np.zeros((len(idx_seleksi),gen))
        for i in range(len(idx_seleksi)):
            selected[i,:]=pop[idx_seleksi[i],:]

        #print('Individu yang terseleksi adalah ',selected)


    #persilangan
    #peluangsilang = float(input('Input peluang persilangan : '))
        jum_silang = int(math.ceil((npop-len(selected))/2))
        hasil_persilangan=np.zeros((int(2*jum_silang),gen))
        for i in range(jum_silang):
            indeks1 = np.random.randint(0,len(selected)-1)
            indeks2 = np.random.randint(0,len(selected)-1)
        
        #rs = np.random.uniform(low = 0, high = 1,size=(1,))
        
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
        

        #print('Hasil persilangan individu terseleksi adalah ',hasil_persilangan)

    #populasi baru
        pop_baru = np.zeros((int(len(selected)+(2*jum_silang)),gen))

        for i in range(int(len(selected))):
            pop_baru[i,] = selected[i,]
        
        for i in range(jum_silang):
            pop_baru[i+int(len(selected)),]=hasil_persilangan[i,]
            pop_baru[i+int(len(selected))+jum_silang,]=hasil_persilangan[i+jum_silang,]

        print('\n')
        print('Bandwidth baru : ',pop_baru)
        print('\n')
    #mutasi
        

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
        print('Hasil mutasi bandwidth : ',pop_baru)
        print('\n')
        pop = pop_baru
        npop = len(pop) #####salah satu pengaturannya sudah di sini
        generasi = generasi+1

        if generasi == ngenerasi:
            band = pd.DataFrame(pop)
            print('\n')
            lokasi_bandwidth = input('Input nama file untuk menyimpan bandwidth optimal (contoh: nama-file.csv) : ')
            print('\n')
            band.to_csv(direktori+lokasi_bandwidth)
            print('\n')
            print('Proses pencarian bandwidth optimal dengan Algoritma Genetika sudah selesai. Silahkan lihat list bandwidth optimal di ', direktori+lokasi_bandwidth)
            print('\n')
            
        print('\n')
        print('Mohon menunggu, data prediksi sedang dikalkulasi berdasarkan bandwidth optimal...')
        print('\n')

    ######## cetak grafik perbandingan antara prediksi dengan data asli ##########

    ##### ambil data bandwidth ######

    band = pd.read_csv(direktori+lokasi_bandwidth)
    band = np.array(band)
    h = np.array(band[0,1:len(band[1,:])])

    ######## perhitungan hasil prediksi ##############

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

    #print('H = ',H)
    #print('trace H = ',np.matrix.trace(np.matrix(H)))

    mh0 = np.mat(H)*np.transpose(np.mat(yk))

    

    input('Pause.... <ENTER> untuk lanjut ke load gambar')

    print('\n')

    print('Load plot.....')

    print('\n')


    #### Plot data dan Prediksi ####

    plt.plot(yk, '-o', label='Data asli')
    plt.plot(mh0, '-.', label='Prediksi')
    plt.legend()
    plt.show()

    hasil = pd.DataFrame(mh0)
    print('\n')
    lokasi_hasil = input('Input nama file untuk menyimpan hasil prediksi (contoh: nama-file.csv) : ')
    print('\n')
    hasil.to_csv(direktori+lokasi_hasil)

    henti_sub = 0
    while henti_sub < 1:
        print('\n')
        ins = input('Apakah Anda ingin mengulangi proses perhitungan? (y/n) ')
        print('\n')
        if ins == 'y':
            henti_sub = 1
            henti = 0
        elif ins == 'n':
            henti_sub = 1
            henti = 1
        else:
            print('\n')
            print('Pilihan yang Anda pilih tidak tersedia.')
            input('Silahkan tekan <ENTER> untuk memilih ulang.')
            print('\n')
            henti_sub = 0


print('\n')
print('Terima Kasih')
print('\n')
input('Tekan ENTER untuk keluar')

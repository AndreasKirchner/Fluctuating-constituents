using Plots
using DelimitedFiles
using Distributions
using LaTeXStrings

function do_pears(name,bins)
    data=readdlm(name)
    data=data[sortperm(data[1:end,4],rev=true),:]
    Nbins=bins
    @show size(data,1)
    Nev_per_bin=Int(floor(size(data,1)/Nbins))
    borders=0:1:Nbins-1

    pears2=map(cc->pearson_corr(data[:,6][cc*Nev_per_bin+1:(cc+1)*Nev_per_bin].^2,data[:,14][cc*Nev_per_bin+1:(cc+1)*Nev_per_bin]),borders)
    pears3=map(cc->pearson_corr(data[:,7][cc*Nev_per_bin+1:(cc+1)*Nev_per_bin].^2,data[:,14][cc*Nev_per_bin+1:(cc+1)*Nev_per_bin]),borders)
    pears4=map(cc->pearson_corr(data[:,8][cc*Nev_per_bin+1:(cc+1)*Nev_per_bin].^2,data[:,14][cc*Nev_per_bin+1:(cc+1)*Nev_per_bin]),borders)
    #kurt=map(cc->kurtosis(Float64.(data[:,obs][cc*Nev_per_bin+1:(cc+1)*Nev_per_bin])),borders)
    #skew=map(cc->skewness(Float64.(data[:,obs][cc*Nev_per_bin+1:(cc+1)*Nev_per_bin])),borders)
    #std=map(cc->Distributions.std(Float64.(data[:,obs][cc*Nev_per_bin+1:(cc+1)*Nev_per_bin])),borders)
    #Mean=map(cc->Distributions.mean(Float64.(data[:,obs][cc*Nev_per_bin+1:(cc+1)*Nev_per_bin])),borders)
    #return [kurt,skew,std,Mean]
    return [pears2,pears3,pears4]
end

function do_cums(name,bins,obs)
    data=readdlm(name)
    data=data[sortperm(data[1:end-1,4],rev=true),:]
    Nbins=bins
    Nev_per_bin=Int(floor(size(data,1)/Nbins))
    borders=0:1:Nbins-1
    kurt=map(cc->kurtosis(Float64.(data[:,obs][cc*Nev_per_bin+1:(cc+1)*Nev_per_bin])),borders)
    skew=map(cc->skewness(Float64.(data[:,obs][cc*Nev_per_bin+1:(cc+1)*Nev_per_bin])),borders)
    std=map(cc->Distributions.std(Float64.(data[:,obs][cc*Nev_per_bin+1:(cc+1)*Nev_per_bin])),borders)
    Mean=map(cc->Distributions.mean(Float64.(data[:,obs][cc*Nev_per_bin+1:(cc+1)*Nev_per_bin])),borders)
    return [Mean,std,skew,kurt]
end

NumCen=100

function do_eps_Ne(obs,NumCen)
    v2m1mean, v2m1std, v2m1skew, v2m1kurt = do_cums("trento_Ne_big_m1.dat",NumCen,obs)
    v2m2mean, v2m2std, v2m2skew, v2m2kurt = do_cums("trento_Ne_big_m2.dat",NumCen,obs)
    v2m3mean, v2m3std, v2m3skew, v2m3kurt = do_cums("trento_Ne_big_m3.dat",NumCen,obs)
    v2m4mean, v2m4std, v2m4skew, v2m4kurt = do_cums("trento_Ne_big_m4.dat",NumCen,obs)
    v2m5mean, v2m5std, v2m5skew, v2m5kurt = do_cums("trento_Ne_big_m5.dat",NumCen,obs)
    ind=obs-3
    writedlm("v"*string(ind)*"_all_Ne.dat",[v2m1mean, v2m1std, v2m1skew, v2m1kurt,v2m2mean, v2m2std, v2m2skew, v2m2kurt,v2m3mean, v2m3std, v2m3skew, v2m3kurt,v2m4mean, v2m4std, v2m4skew, v2m4kurt,v2m5mean, v2m5std, v2m5skew, v2m5kurt])
end

function do_eps_O(obs,NumCen)
    v2m1mean, v2m1std, v2m1skew, v2m1kurt = do_cums("trento_O_big_m1.dat",NumCen,obs)
    v2m2mean, v2m2std, v2m2skew, v2m2kurt = do_cums("trento_O_big_m2.dat",NumCen,obs)
    v2m3mean, v2m3std, v2m3skew, v2m3kurt = do_cums("trento_O_big_m3.dat",NumCen,obs)
    v2m4mean, v2m4std, v2m4skew, v2m4kurt = do_cums("trento_O_big_m4.dat",NumCen,obs)
    v2m5mean, v2m5std, v2m5skew, v2m5kurt = do_cums("trento_O_big_m5.dat",NumCen,obs)
    ind=obs-3
    writedlm("v"*string(ind)*"_all_O.dat",[v2m1mean, v2m1std, v2m1skew, v2m1kurt,v2m2mean, v2m2std, v2m2skew, v2m2kurt,v2m3mean, v2m3std, v2m3skew, v2m3kurt,v2m4mean, v2m4std, v2m4skew, v2m4kurt,v2m5mean, v2m5std, v2m5skew, v2m5kurt])
end

for i in 5:8
    do_eps_Ne(i,NumCen)
    do_eps_O(i,NumCen)
end
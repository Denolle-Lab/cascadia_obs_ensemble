### D-B048: GrowClust3D driver (from GrowClust3D.jl examples/run_growclust3D.jl, git 592f6179, single process,
### bootstraps on threads). Changes against the example: the algorithm constants are read from a "key value" file
### (ARGS[2], written by b5_growclust.py) instead of being hard-coded; nothing else.
###   julia -t N b5_growclust_run.jl control.inp aux.txt

using Printf
using DataFrames
using Random
using Dates
using GrowClust3D

aux = Dict{String,String}()
for line in eachline(ARGS[2])
    s = split(strip(line))
    if length(s) >= 2 && s[1][1] != '#'
        aux[s[1]] = s[2]
    end
end
const distmax = parse(Float64, aux["distmax"])
const distmax2 = parse(Float64, aux["distmax2"])
const hshiftmax = parse(Float64, aux["hshiftmax"])
const vshiftmax = parse(Float64, aux["vshiftmax"])
const rmedmax = parse(Float32, aux["rmedmax"])
const maxlink = parse(Int64, aux["maxlink"])
const nbeststa = parse(Int64, aux["nbeststa"])
const nupdate = 10000
const boxwid = parse(Float64, aux["boxwid"])
const nit = parse(Int64, aux["nit"])
const irelonorm = parse(Int64, aux["irelonorm"])
const tdifmax = Float32(30.0)
const torgdifmax = parse(Float32, aux["torgdifmax"])
const iseed = 0
const vzmodel_type = 1
const shallowmode = "flat"
const degkm = 111.1949266
const erad = 6371.0

println("\nReading input file: ", ARGS[1])
inpD = read_gcinp(ARGS[1])
if inpD["vpvs_factor"] < 0.01
    inpD["vpvs_factor"] = sqrt(3.0)
end
input_ok = check_gcinp(inpD)
if !input_ok
    println("ERROR: FIX INPUT PARAMETERS")
    exit(1)
end
for fkey in collect(keys(inpD))
    if startswith(fkey, "fout") && inpD[fkey] != "NONE"
        mkpath(join(split(inpD[fkey], "/")[1:end-1], "/"))
    end
end
params_ok = check_auxparams(hshiftmax, vshiftmax, rmedmax, boxwid, nit, irelonorm, vzmodel_type)
if !params_ok
    println("ERROR: FIX AUXILIARY PARAMETERS")
    exit(1)
end
print_input(inpD)

qdf = read_evlist(inpD["fin_evlist"], inpD["evlist_fmt"])
plat0, plon0 = inpD["lat0"], inpD["lon0"]
plat1, plat2 = inpD["latp1"], inpD["latp2"]
sdf = read_stlist(inpD["fin_stlist"], inpD["stlist_fmt"])
proj = setup_projection(inpD, plon0, plat0, plat1, plat2)
iproj = inv(proj)
sX4, sY4 = lonlat2xypos(sdf.slon, sdf.slat, inpD["rotANG"], proj)
sdf[!, :sX4] .= sX4
sdf[!, :sY4] .= sY4
qX4, qY4 = lonlat2xypos(qdf.qlon, qdf.qlat, inpD["rotANG"], proj)
qdf[!, :qX4] .= qX4
qdf[!, :qY4] .= qY4
xdf = read_xcordata_proj(inpD, qdf[!, [:qix, :qid, :qX4, :qY4]], sdf[!, [:sta, :sX4, :sY4]])
nbad = sum(abs.(xdf.tdif) .> tdifmax)
if nbad > 0
    println("Error: differential times larger than tdifmax=$tdifmax")
    exit(1)
end
usta = unique(xdf[!, "sta"])
nstaU = length(usta)
println("$nstaU unique stations used:\n", usta, "\n")
xdf[!, :itab] .= Int16(0)
staIDX = Dict(zip(usta, 1:length(usta)))
for ii = 1:nrow(xdf)
    xdf[ii, :itab] = staIDX[xdf[ii, :sta]] + nstaU * (xdf[ii, :iphase] - 1)
end
subset!(sdf, :sta => ByRow(x -> x in usta))
maxSR = maximum(xdf.sdist)
ttLIST = make_nllgrid_tables(inpD, maxSR, usta, shallowmode)
const ttTABs = [deepcopy(xx) for xx in ttLIST]
@printf("min and max event depth: %.3fkm %.3fkm\n", minimum(qdf.qdep), maximum(qdf.qdep))
if maximum(qdf.qdep) > inpD["tt_zmax"]
    println("ERROR: max event depth > max table depth")
    exit(1)
end

const nq = Int32(nrow(qdf))
const rmincut, rmsmax = inpD["rmincut"], inpD["rmsmax"]
const tt_ndim, rotANG = inpD["tt_ndim"], inpD["rotANG"]
revids = qdf[:, :qid]
rlats, rlons, rdeps = qdf[:, :qlat], qdf[:, :qlon], qdf[:, :qdep]
rXs, rYs = qdf[:, :qX4], qdf[:, :qY4]
rorgs = zeros(Float32, nq)
rcids = Vector{Int32}(1:nq)
if inpD["nboot"] > 0
    blatM = repeat(qdf.qlat, 1, inpD["nboot"])
    blonM = repeat(qdf.qlon, 1, inpD["nboot"])
    bdepM = repeat(qdf.qdep, 1, inpD["nboot"])
    borgM = zeros(Float32, (nq, inpD["nboot"]))
    bnbM = repeat(Vector{Int32}(1:nq), 1, inpD["nboot"])
else
    blatM, blonM, bdepM, borgM, bnbM = Nothing, Nothing, Nothing, Nothing, Nothing
end
xdf00 = select(xdf, [:qix1, :qix2, :sX4, :sY4, :tdif, :itab, :rxcor, :igood])
const nxc = Int32(nrow(xdf00))
const ixc = Vector{Int32}(1:nxc)

println("\nStarting relocation estimates, nthread=", Threads.nthreads())
@time Threads.@threads for ib in 0:inpD["nboot"]
    Random.seed!(iseed + ib)
    if ib > 0
        isamp = sort(sample(ixc, nxc, replace=true))
        rxdf = xdf00[isamp, :]
    else
        rxdf = xdf00
    end
    rxdf[!, :ixx] = Vector{Int32}(1:nxc)
    bpdf = combine(groupby(rxdf[!, Not([:sX4, :sY4, :itab, :tdif])], [:qix1, :qix2]),
        :rxcor => (x -> topNmeanpad(x, nbeststa, pad=rmincut / 2.0)) => :rfactor,
        :ixx => first => :ix1, :ixx => last => :ix2, :igood => sum => :ngood)
    if ib > 0
        bpdf = bpdf[bpdf.ngood.>=inpD["ngoodmin"]-2, [:qix1, :qix2, :rfactor, :ix1, :ix2]]
    else
        select!(bpdf, [:qix1, :qix2, :rfactor, :ix1, :ix2])
    end
    sort!(bpdf, :rfactor, rev=true)
    brXs, brYs, brdeps, brorgs, brcids, bnb = clustertree_3D(
        bpdf.qix1, bpdf.qix2, bpdf.ix1, bpdf.ix2, rxdf.tdif, rxdf.itab,
        qdf.qX4, qdf.qY4, qdf.qdep, ttTABs, nit, boxwid, irelonorm, rmsmax,
        rmedmax, distmax, distmax2, hshiftmax, vshiftmax, torgdifmax, nupdate, maxlink)
    brlons, brlats = xypos2latlon(brXs, brYs, rotANG, iproj)
    if ib > 0
        blatM[:, ib] .= brlats
        blonM[:, ib] .= brlons
        bdepM[:, ib] .= brdeps
        borgM[:, ib] .= brorgs
        bnbM[:, ib] .= bnb
    else
        rXs .= brXs
        rYs .= brYs
        rlats .= brlats
        rlons .= brlons
        rdeps .= brdeps
        rorgs .= brorgs
        rcids .= brcids
    end
end

rnbranch, rcids, tnbranch, tlats, tlons, tdeps, torgs = make_clustertree(rlons, rlats, rdeps, rorgs, rcids, qdf.qix)
rdf = DataFrame("enum" => qdf.qix, "evid" => revids, "mag" => qdf.qmag,
    "rlat" => rlats, "rlon" => rlons, "rdep" => rdeps, "rtim" => rorgs,
    "rcid" => rcids, "rnb" => rnbranch, "rX" => rXs, "rY" => rYs,
    "qlat" => qdf.qlat, "qlon" => qdf.qlon, "qdep" => qdf.qdep)
println("Relocated: ", sum(rdf.rnb .> 1))
rdf[!, :rot] = qdf[!, :qotime] .+ Nanosecond.(round.(rorgs * 1.0e9))
npp, nss, rmsP, rmsS, msresP, msresS, qrmsP, qrmsS, qndiffP, qndiffS, qnpair = compute_misfits(inpD, xdf, rdf, ttTABs)
boot_madH, boot_madZ, boot_madT, boot_stdH, boot_stdZ, boot_stdT, boot_nbL, boot_nbM, boot_nbH =
    compute_bootstats(inpD, degkm, rdf, bnbM, blonM, blatM, bdepM, borgM)
write_cat(inpD, rdf, qnpair, qndiffP, qndiffS, qrmsP, qrmsS, boot_madH, boot_madZ, boot_madT)
if !(inpD["fout_clust"] in ["none", "None", "NONE"])
    write_clust(inpD, rdf, tnbranch, tlats, tlons, tdeps, torgs, boot_madH, boot_madT, boot_madZ)
end
if (inpD["nboot"] > 1) & (!(inpD["fout_boot"] in ["none", "None", "NONE"]))
    write_boot(inpD, rdf, boot_madH, boot_madT, boot_madZ, boot_nbH, boot_nbL, boot_nbM, boot_stdH, boot_stdT,
        boot_stdZ, blatM, blonM, bdepM, bnbM)
end
write_log(inpD, [ARGS[1], distmax, distmax2, hshiftmax, vshiftmax, rmedmax],
    [nq, sum(rdf.rnb .> 1), qnpair, npp, nss, rmsP, rmsS, msresP, msresS], tnbranch)
@printf("\nCompleted task: b5_growclust_run.jl\n")


# Relevant base files:
# darray.jl
# multi.jl

function memstats()
  println(run(`ps -C julia -o "%P" -o "%mem" -o "%CPU"`))
end


function remote_vars()
  for w in workers()
    println(remotecall_fetch(w, whos))
  end
end

# This is like @everywhere but only runs on a particular process.
macro runat(p, ex)
  quote
    remotecall_fetch($p, ()->(eval(Main,$(Expr(:quote,ex))); nothing))
  end
end


# Like @everywhere but for remote workers.
macro everywhereelse(ex)
    quote
        @sync begin
            for w in workers()
                @async remotecall_fetch(w, ()->(eval(Main,$(Expr(:quote,ex))); nothing))
            end
        end
    end
end


@everywhereelse println(myid())
@everywhere println(myid())


# A working example with a big object:
mat = rand(int(1e4), int(1e4));

addprocs(2)
memstats()

nw = length(workers())
col_cuts = iround(linspace(1, size(mat)[2] + 1, nw + 1))
col_ranges = map(i -> col_cuts[i]:col_cuts[i + 1] - 1, 1:nw)


# This works:
@everywhereelse rr = RemoteRef(1)
rr_array = [ remotecall_fetch(w, () -> rr) for w in workers() ]
for iw=1:nw
  put!(rr_array[iw], mat[:,col_ranges[iw]])
end

@everywhereelse local_mat = fetch(rr);
memstats()

@everywhereelse println(size(local_mat))
@everywhereelse local_mat_sum = sum(local_mat)
remote_sums = [ remotecall_fetch(w, () -> local_mat_sum) for w in workers() ]
sum(remote_sums) == sum(mat)









######################################
# DArray stuff

# This doubles the memory usage on 1 and does copy the parts
# of the array over to the other processes.
dmat = distribute(mat);
gc()
memstats()

# This hangs. :(
#sum(dmat)

# This works because it resolved dmat on process 1
remotecall_fetch(2, localindexes, dmat)

# This does not because dmat is not an object on process 2
#@runat 2 localindexes(dmat)

# The DArray object is just a container of remoterefs and indices.
dmat1 = fetch(dmat.chunks[1]);
dmat2 = fetch(dmat.chunks[2]);
size(dmat1)
size(dmat2)

# So distributing the DArray object takes no memory.
@everywhere gc()
memstats()
@everywhere rr = RemoteRef(1)
rr_array = [ remotecall_fetch(w, () -> rr) for w in workers() ]
for worker_rr in rr_array
  put!(worker_rr, dmat)
end
@runat 2 dmat = fetch(rr)
@runat 3 dmat = fetch(rr)
memstats()

@everywhere locali = localindexes(dmat)
@everywhere firsti = (first(locali[1]), first(locali[2])[1])
@everywhere lasti = (last(locali[1]), last(locali[2])[1])

# Does each indexing operation prompt a fetch?  It seems like it should from
# the getindex() function for DArrays.  However, after the first reference
# subsequent references are fast even though there is no increase in
# memory.
@everywhere gc()
memstats()
@time println(dmat[firsti...]);
memstats()
@time @runat 2 println(dmat[firsti...]);
@everywhere gc()
memstats()
@time @runat 2 println(dmat[firsti...]);
memstats()
@everywhere gc()
@time @runat 2 println(dmat[lasti...]);
memstats()

# Note that the remote refs have the remote nodes as their origin:
dmat.chunks

# The first fetch on a local node always takes a little longer.
rr = RemoteRef(1)
put!(rr, mat);
@time fetch(rr)[1, 1]
@time fetch(rr)[1, 1]

# But fetches from a remote node take longer each time.
@runat 2 rr = remotecall_fetch(1, () -> rr)
@runat 2 @time fetch(rr)[1, 1]
@runat 2 @time fetch(rr)[1, 1]

# But how is it that process 3 can quickly access elements outside its
# range?  getindex is calling sub(), and that seems like a deep hold to
# go down.
@runat 3 println(localindexes(dmat))
@time @runat 3 println(dmat[1, 1]);
@time @runat 3 println(dmat[1, 1]);
@time @runat 3 println(dmat[2, 2]);
@time @runat 3 println(dmat[2, 2]);

@time sum(mat)

# Why?  This local function defn doesn't work for some reason.

# find which piece holds index (I...)
function locate(d::DArray, I::Int...)
    ntuple(ndims(d), i->searchsortedlast(d.cuts[i], I[i]))
end

i = map(x -> 1:x, size(dmat))
I = ind2sub(size(dmat), 1)
chidx = locate(dmat, I...)
chunk = d.chunks[chidx...]
idxs = d.indexes[chidx...]
localidx = ntuple(ndims(d), i->(I[i]-first(idxs[i])+1))


@everywhere rr = RemoteRef(1)
rr_array = [ remotecall_fetch(worker, () -> rr) for worker in workers() ]
memstats()

for worker_rr in rr_array
  put!(worker_rr, mat)
end
memstats()

for worker_rr in rr_array
  println(worker_rr.where, ": ", fetch(worker_rr)[1:5, 1:5])
end
memstats()

# Now this actually copies the array over.  Though for some reason it
# also increases the memory usage of the first process.
@runat 2 mat11 = fetch(rr)[1,1]
memstats()

# Making a local copy of the whole array uses as much memory as saving
# just he first element.
@runat 2 mat_worker = fetch(rr)
memstats()

# But the extra memory is freed with garbage collection.
@everywhere gc()
memstats()

# Confirm that the copy on worker two is in fact a local copy
@runat 2 mat_worker[1,1]= -20.
@runat 2 rr2 = RemoteRef(2)
@runat 2 put!(rr2, mat_worker)
rr2 = remotecall_fetch(2, () -> rr2)
mat_worker = fetch(rr2);
mat_worker[1, 1]
mat[1, 1]
memstats()

# Note that this only runs up the memory usage on process 2.
@runat 2 num_big_mats = 5
@runat 2 big_mats = Array(Any, num_big_mats);
@runat 2 for i = 1:num_big_mats
            big_mats[i] = rand(int(1e4), int(1e4))
          end
memstats()

# Copy from process 2 to process 1.
@runat 2 rr = RemoteRef(2)
@runat 2 put!(rr, big_mats[1])
rr = remotecall_fetch(2, () -> rr)
big_mat_1 = fetch(rr);
memstats()

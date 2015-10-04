

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


# A working example with a big object:
mat = rand(int(1e4), int(1e4));
mat2 = rand(int(1e4), int(1e4));

addprocs(2)
memstats()

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

# Now this actually copies the array over.  Note that extra memory is allocated
# during the copy but is freed by gc().
fetch_time1 = time()
@runat 2 mat11 = fetch(rr)[1,1]
fetch_time1 = time() - fetch_time1
memstats()
gc()
memstats()

# Subsequent fetches take just as much time and allocate just as much memory
# before gc().  It is evidently re-fetching each time.
fetch_time2 = time()
@runat 2 mat12 = fetch(rr)[1,2]
fetch_time2 = time() - fetch_time2
memstats()

# Making a local copy of the whole array.
@runat 2 mat_worker = fetch(rr)
memstats()

# Again, the extra memory is freed with garbage collection.
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

# Subsequent fetches doesn't pick up changes in the immutable objects
@runat 2 rr = RemoteRef(1)
rr = remotecall_fetch(2, () -> rr)
myx = 1.23
put!(rr, myx)
@runat 2 println(fetch(rr))
myx += 1.45
@runat 2 println(fetch(rr))

# But it does in mutable ones
@runat 2 rr = RemoteRef(1)
rr = remotecall_fetch(2, () -> rr)
myxvec = [ 1.23 ]
put!(rr, myxvec)
@runat 2 println(fetch(rr))
myxvec[1] += 1.45
@runat 2 println(fetch(rr))

# Note that this doesn't work right when the remoteref source and
# destination don't match.  For example, to dynamically update
# a variable in process 2 as seen by process 1, the RemoteRef must
# be initialized on process 1.  TODO: document this.

# Compare.  The destination of a new RemoteRef is always the current process.
RemoteRef(1) # Won't work for communicating from 1
@everywhereelse println(RemoteRef(1)) # will work for communicating from 1


# Warning: workers() does not return the same order everywhere.
@everywhere println(workers())



#############
# Some of the initial flailing

# This doesn't work for assignment.
for worker_rr in rr_array
  @spawnat worker_rr.where mat2 = fetch(worker_rr)
  @spawnat worker_rr.where y = 1.234
end
memstats()
remote_vars()

# This works though
@everywhere x = 123

# Doesn't work, maybe something weird with keywords:
# @runat(2, y=1.23)

mat = rand(2, 2);

rr = RemoteRef(1)
put!(rr, mat)

# Not defined
@everywhere fetch(rr)

# This works!
@everywhere x = fetch(RemoteRef(1, 1, 4))

# This does not:
macro dereference_rr(rr)
  :(:(RemoteRef(eval(rr.where), eval(rr.whence), eval(rr.id))))
end

rr_symb = @dereference_rr(rr)
@everywhere println(fetch))

# How about this?  This works great.
@everywhere rr = RemoteRef(1)
rr_array = [ remotecall_fetch(worker, () -> rr) for worker in workers() ]

# Note that you cannot rerun this -- once you have run put! on a RemoteRef,
# you cannot run it again.
i = 10.0
for worker_rr in rr_array
  put!(worker_rr, i)
  i += 1
end

for worker_rr in rr_array
  println(worker_rr.where, ": ", fetch(worker_rr))
end

# This does not seem to clear the meory for some reason
@runat 2 big_mats = 0
@runat 2 mat_worker = 0
mat = 0
big_mat_1 = 0
@everywhere gc()



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

# Now this actually copies the array over.  Though for some reason it
# also increases the memory usage of the first process.  Maybe ps
# counts the memory used by sub-processes?
@runat 2 mat_worker = fetch(rr)
memstats()

@runat 2 mat_worker[1,1]= -20.
@runat 2 rr2 = RemoteRef(2)
@runat 2 put!(rr2, mat_worker)
rr2 = remotecall_fetch(2, () -> rr2)
mat_worker = fetch(rr2);
mat_worker[1, 1]
memstats()

# Note that this only runs up the memory usage on process 2
@runat 2 num_big_mats = 5
@runat 2 big_mats = Array(Any, num_big_mats);
@runat 2 for i = 1:num_big_mats
            big_mats[i] = rand(int(1e4), int(1e4))
          end
memstats()









#############
# Stuff that doesn't work

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


# A working example with a big thing:
mat = rand(int(1e4), int(1e5));
@everywhere rr = RemoteRef(1)
rr_array = [ remotecall_fetch(worker, () -> rr) for worker in workers() ]

tic()
for worker_rr in rr_array
  put!(worker_rr, mat)
end
toc()

tic()
for worker_rr in rr_array
  println(worker_rr.where, ": ", fetch(worker_rr)[1:5, 1:5])
end
toc()

tic()
for worker_rr in rr_array
  println(worker_rr.where, ": ", fetch(worker_rr)[1:5, 1:5])
end
toc()

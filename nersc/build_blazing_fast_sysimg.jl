#!julia
if length(ARGS) < 2
    println("Usage: build_balzing_fast_sysimage.sh [--opt-only] <mcpu> <PRECOMPILE_REQUEST>")
end
opt_only = false
if ARGS[1] == "--opt-only"
    opt_only = true
    shift!(ARGS)
end
if length(ARGS) > 2
    println("ERROR: Too many arguments. Superflous arguments were $(ARGS[3:end]). Did you forget quotes?")
    exit(1)
end
mcpu = ARGS[1]
if !(mcpu in ["haswell","knl"])
    println("ERROR: <mcpu> must be one of haswell,knl")
    exit(1)
end
request = ARGS[2]

LLVM_BIN_DIR = joinpath(JULIA_HOME,"..","tools")
JULIA_LIB_DIR = joinpath(JULIA_HOME,"..","lib")
JULIA_BASE_SYS_IMG = unsafe_string(Base.JLOptions().image_file)
ENV["JULIA_NUM_THREADS"]=1
full_request = "Base.Sys.__init__(); Base.Random.__init__(); Base.LinAlg.__init__(); $request"
if !opt_only
run(`julia --mcpu=$mcpu --output-bc sys-all.bc --sysimage $JULIA_BASE_SYS_IMG --depwarn=no --startup-file=no -O3 --eval $full_request`)
end
run(pipeline(`$("$LLVM_BIN_DIR/llvm-extract") --recursive -rfunc .\*elbo_likelihood.\* -rfunc .\*first_quad_form.\* -rfunc .\*propagate_derivatives.\* sys-all.bc`, "hotspot.bc"))
run(pipeline(`$("$LLVM_BIN_DIR/llvm-extract") --delete --recursive -rfunc .\*elbo_likelihood.\* -rfunc .\*first_quad_form.\* -rfunc .\*propagate_derivatives.\* sys-all.bc`, "residual.bc"))
run(pipeline(`$("$LLVM_BIN_DIR/opt") -inline-threshold=3000 -inline hotspot.bc`, "hotspot-inlined.bc"))
run(pipeline(`$("$LLVM_BIN_DIR/llvm-extract") -S -rfunc .\*populate_gal_fsm.\* hotspot-inlined.bc`,"gal_fsm.ll"))
run(pipeline(`$("$LLVM_BIN_DIR/opt") -strip-debug -memdep-block-scan-limit=10000 -tbaa -basicaa -scev-aa -scoped-noalias -S -mcpu $mcpu -licm -gvn -dse -instcombine -sroa -instcombine -licm -instcombine -newgvn -licm -instcombine -lcssa -dse gal_fsm.ll`,"gal_fsm2.ll"))
# Poor man's LLVM pass - Can be done properly - no time
run(`sed -i 's/fadd/fadd fast/g' gal_fsm2.ll`)
run(`sed -i 's/fsub/fsub fast/g' gal_fsm2.ll`)
run(`sed -i 's/fmul/fmul fast/g' gal_fsm2.ll`)
run(pipeline(`$("$LLVM_BIN_DIR/opt") -tbaa -basicaa -scev-aa -scoped-noalias -mcpu $mcpu -vector-library=SVML -loop-vectorize -instcombine -S gal_fsm2.ll`,"gal_fsm_vectorized.ll"))
run(pipeline(`$("$LLVM_BIN_DIR/llvm-extract") --delete -S -rfunc .\*populate_gal_fsm.\* hotspot.bc`,"hotspot-rest.bc"))
run(pipeline(`$("$LLVM_BIN_DIR/opt") -tbaa -basicaa -scev-aa -scoped-noalias -mcpu $mcpu -licm -gvn -slp-vectorizer -instcombine -loop-vectorize -instcombine hotspot-rest.bc`,"hotspot-rest-opt.bc"))
#~/llvm-debug-build/bin/opt -tbaa -basicaa -scev-aa -scoped-noalias -mcpu $mcpu -licm -gvn -loop-vectorize -instcombine residual.bc > residual-opt.bc
run(`$("$LLVM_BIN_DIR/llc") residual.bc -fp-contract=fast -mcpu=$mcpu -relocation-model=pic -filetype=obj -o residual.o`)
run(`$("$LLVM_BIN_DIR/llc") hotspot-rest-opt.bc -fp-contract=fast -mcpu=$mcpu -relocation-model=pic -filetype=obj -o hotspot.o`)
run(`$("$LLVM_BIN_DIR/llc") gal_fsm_vectorized.ll -fp-contract=fast -mcpu=$mcpu -relocation-model=pic -filetype=obj -o gal_fsm.o`)
run(`cc -shared -o sys-all.so residual.o hotspot.o gal_fsm.o`)

#=
JULIA_NUM_THREADS=1 julia --mcpu=$mcpu --output-bc sys-all.bc --sysimage $JULIA_BASE_SYS_IMG --depwarn=no --startup-file=no -O3 --eval 'Base.Sys.__init__(); Base.Random.__init__(); Base.LinAlg.__init__(); include("../benchmark_one_light_source.jl")'
$LLVM_BIN_DIR/llvm-extract --recursive -rfunc .*elbo_likelihood.* -rfunc .*first_quad_form.* -rfunc .*propagate_derivatives.* sys-all.bc > hotspot.bc
$LLVM_BIN_DIR/llvm-extract --delete --recursive -rfunc .*elbo_likelihood.* -rfunc .*first_quad_form.* -rfunc .*propagate_derivatives.* sys-all.bc > residual.bc
#~/julia/usr/tools/llc residual.bc -fp-contract=fast -mcpu=$mcpu -relocation-model=pic -filetype=obj -o residual.o
#~/julia/usr/tools/llc hotspot.bc -fp-contract=fast -mcpu=$mcpu -relocation-model=pic -filetype=obj -o hotspot.o
#cc -shared -o sys-all.so residual.o hotspot.o -L ~/julia/usr/lib -ljulia
$LLVM_BIN_DIR/opt -inline-threshold=3000 -inline hotspot.bc > hotspot-inlined.bc
$LLVM_BIN_DIR/llvm-extract -S -rfunc .*populate_gal_fsm.* hotspot-inlined.bc > gal_fsm.ll
$LLVM_BIN_DIR/opt -strip-debug -memdep-block-scan-limit=10000 -tbaa -basicaa -scev-aa -scoped-noalias -S -mcpu $mcpu -licm -gvn -dse -instcombine -sroa -instcombine -licm -instcombine -newgvn -licm -instcombine -lcssa -dse gal_fsm.ll > gal_fsm2.ll
# Poor man's LLVM pass - Can be done properly - no time
sed -i 's/fadd/fadd fast/g' gal_fsm2.ll
sed -i 's/fsub/fsub fast/g' gal_fsm2.ll
sed -i 's/fmul/fmul fast/g' gal_fsm2.ll
$LLVM_BIN_DIR/opt -tbaa -basicaa -scev-aa -scoped-noalias -mcpu $mcpu -vector-library=SVML -loop-vectorize -instcombine -S gal_fsm2.ll > gal_fsm_vectorized.ll
$LLVM_BIN_DIR/llvm-extract --delete -S -rfunc .*populate_gal_fsm.* hotspot.bc > hotspot-rest.bc
$LLVM_BIN_DIR/opt -tbaa -basicaa -scev-aa -scoped-noalias -mcpu $mcpu -licm -gvn -slp-vectorizer -instcombine -loop-vectorize -instcombine hotspot-rest.bc > hotspot-rest-opt.bc
#~/llvm-debug-build/bin/opt -tbaa -basicaa -scev-aa -scoped-noalias -mcpu $mcpu -licm -gvn -loop-vectorize -instcombine residual.bc > residual-opt.bc
$LLVM_BIN_DIR/llc residual.bc -fp-contract=fast -mcpu=$mcpu -relocation-model=pic -filetype=obj -o residual.o
$LLVM_BIN_DIR/llc hotspot-rest-opt.bc -fp-contract=fast -mcpu=$mcpu -relocation-model=pic -filetype=obj -o hotspot.o
$LLVM_BIN_DIR/llc gal_fsm_vectorized.ll -fp-contract=fast -mcpu=$mcpu -relocation-model=pic -filetype=obj -o gal_fsm.o
cc -shared -o sys-all.so residual.o hotspot.o gal_fsm.o
=#


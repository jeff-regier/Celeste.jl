using BinDeps

@BinDeps.setup

version = "1.0.2"
url = "https://github.com/kbarbary/sep/archive/v$(version).tar.gz"

libsep = library_dependency("libsep")
println(libsep)
downloadsdir = BinDeps.downloadsdir(libsep)
libdir = BinDeps.libdir(libsep)
srcdir = BinDeps.srcdir(libsep)
if is_apple()
    libfilename = "libsep.dylib"
elseif is_unix()
    libfilename = "libsep.so"
end

# Unix
prefix = joinpath(BinDeps.depsdir(libsep), "usr")
provides(Sources, URI(url), libsep, unpacked_dir="sep-$(version)")
provides(BuildProcess,
         (@build_steps begin
             GetSources(libsep)
             @build_steps begin
                 ChangeDirectory(joinpath(srcdir, "sep-$(version)"))
                 FileRule(joinpath(libdir, libfilename),
                          @build_steps begin
                              `make`
                              `make PREFIX=$(prefix) install`
                          end)
             end
          end), libsep, os = :Unix)


@BinDeps.install Dict(:libsep => :libsep)

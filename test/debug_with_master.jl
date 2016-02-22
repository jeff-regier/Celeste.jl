module Debug
  # A path to a different github client on the master branch.
  git_repo_loc = "/home/rgiordan/Documents/git_repos"
  master_path =
    joinpath(git_repo_loc, "debugging_branch_Celeste.jl/CelesteDebug.jl")
  push!(LOAD_PATH, joinpath(master_path, "src"))

  include(joinpath(master_path, "src/Celeste.jl"))
  include(joinpath(master_path, "src/CelesteTypes.jl"))
  include(joinpath(master_path, "src/SampleData.jl"))
  include(joinpath(master_path, "src/KL.jl"))
  include(joinpath(master_path, "src/ElboDeriv.jl"))

end

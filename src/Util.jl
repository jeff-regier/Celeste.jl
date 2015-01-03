# written by Jeffrey Regier
# jeff [at] stat [dot] berkeley [dot] edu

module Util

export matvec222

function matvec222(mat::Matrix, vec::Vector)
	(mat[1,1] * vec[1] + mat[1,2] * vec[2]) * vec[1] + 
			(mat[2,1] * vec[1] + mat[2,2] * vec[2]) * vec[2]
end

end

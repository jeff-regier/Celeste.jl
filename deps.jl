const python = "python"
const libpython = "libpython2.7"
const pyprogramname = bytestring("/usr/bin/python")
const pyversion_build = v"2.7.9"
const PYTHONHOME = bytestring("/usr:/usr")

"True if we are using the Python distribution in the Conda package."
const conda = false

const PyUnicode_AsUTF8String = :PyUnicodeUCS4_AsUTF8String
const PyUnicode_DecodeUTF8 = :PyUnicodeUCS4_DecodeUTF8

const PyString_FromString = :PyString_FromString
const PyString_AsString = :PyString_AsString
const PyString_Size = :PyString_Size
const PyString_Type = :PyString_Type
const PyInt_Type = :PyInt_Type
const PyInt_FromSize_t = :PyInt_FromSize_t
const PyInt_FromSsize_t = :PyInt_FromSsize_t
const PyInt_AsSsize_t = :PyInt_AsSsize_t

const Py_hash_t = Int64

const pyunicode_literals = false

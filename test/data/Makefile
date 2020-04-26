URLBASE := https://data.sdss.org/sas/dr12/boss
RUN := 3900
CAMCOL := 6
FIELD := 269

# strip leading zeros
RUN_STRIPPED := $(shell echo $(RUN) | sed 's/^0*//')
FIELD_STRIPPED := $(shell echo $(FIELD) | sed 's/^0*//')

RUN6 := $(shell printf '%06d' $(RUN_STRIPPED))
FIELD4 := $(shell printf '%04d' $(FIELD_STRIPPED))

SUBDIR2 := $(RUN_STRIPPED)/$(CAMCOL)
SUBDIR3 := $(SUBDIR2)/$(FIELD_STRIPPED)

all : $(SUBDIR3)/fpM-$(RUN6)-u$(CAMCOL)-$(FIELD4).fit \
      $(SUBDIR3)/fpM-$(RUN6)-g$(CAMCOL)-$(FIELD4).fit \
      $(SUBDIR3)/fpM-$(RUN6)-r$(CAMCOL)-$(FIELD4).fit \
      $(SUBDIR3)/fpM-$(RUN6)-i$(CAMCOL)-$(FIELD4).fit \
      $(SUBDIR3)/fpM-$(RUN6)-z$(CAMCOL)-$(FIELD4).fit \
      $(SUBDIR3)/frame-u-$(RUN6)-$(CAMCOL)-$(FIELD4).fits \
      $(SUBDIR3)/frame-g-$(RUN6)-$(CAMCOL)-$(FIELD4).fits \
      $(SUBDIR3)/frame-r-$(RUN6)-$(CAMCOL)-$(FIELD4).fits \
      $(SUBDIR3)/frame-i-$(RUN6)-$(CAMCOL)-$(FIELD4).fits \
      $(SUBDIR3)/frame-z-$(RUN6)-$(CAMCOL)-$(FIELD4).fits \
      $(SUBDIR3)/psField-$(RUN6)-$(CAMCOL)-$(FIELD4).fit \
      $(SUBDIR2)/photoField-$(RUN6)-$(CAMCOL).fits \
      field_extents.fits \
      photoobj

photoobj : $(SUBDIR3)/photoObj-$(RUN6)-$(CAMCOL)-$(FIELD4).fits

field_extents.fits :
	curl -L -O http://portal.nersc.gov/project/dasrepo/celeste/field_extents.fits

$(SUBDIR3)/fpM-$(RUN6)-%$(CAMCOL)-$(FIELD4).fit.gz :
	curl -L --create-dirs -o $(SUBDIR3)/fpM-$(RUN6)-$*$(CAMCOL)-$(FIELD4).fit.gz $(URLBASE)/photo/redux/301/$(RUN_STRIPPED)/objcs/$(CAMCOL)/fpM-$(RUN6)-$*$(CAMCOL)-$(FIELD4).fit.gz
$(SUBDIR3)/fpM-$(RUN6)-%$(CAMCOL)-$(FIELD4).fit : $(SUBDIR3)/fpM-$(RUN6)-%$(CAMCOL)-$(FIELD4).fit.gz
	gunzip < $(SUBDIR3)/fpM-$(RUN6)-$*$(CAMCOL)-$(FIELD4).fit.gz > $(SUBDIR3)/fpM-$(RUN6)-$*$(CAMCOL)-$(FIELD4).fit

$(SUBDIR3)/psField-$(RUN6)-$(CAMCOL)-$(FIELD4).fit :
	curl -L --create-dirs -o $(SUBDIR3)/psField-$(RUN6)-$(CAMCOL)-$(FIELD4).fit $(URLBASE)/photo/redux/301/$(RUN_STRIPPED)/objcs/$(CAMCOL)/psField-$(RUN6)-$(CAMCOL)-$(FIELD4).fit

$(SUBDIR3)/frame-%-$(RUN6)-$(CAMCOL)-$(FIELD4).fits.bz2 :
	curl -L --create-dirs -o $(SUBDIR3)/frame-$*-$(RUN6)-$(CAMCOL)-$(FIELD4).fits.bz2 $(URLBASE)/photoObj/frames/301/$(RUN_STRIPPED)/$(CAMCOL)/frame-$*-$(RUN6)-$(CAMCOL)-$(FIELD4).fits.bz2
$(SUBDIR3)/frame-%-$(RUN6)-$(CAMCOL)-$(FIELD4).fits : $(SUBDIR3)/frame-%-$(RUN6)-$(CAMCOL)-$(FIELD4).fits.bz2
	bunzip2 < $(SUBDIR3)/frame-$*-$(RUN6)-$(CAMCOL)-$(FIELD4).fits.bz2 > $(SUBDIR3)/frame-$*-$(RUN6)-$(CAMCOL)-$(FIELD4).fits

$(SUBDIR3)/photoObj-$(RUN6)-$(CAMCOL)-$(FIELD4).fits :
	curl -L --create-dirs -o $(SUBDIR3)/photoObj-$(RUN6)-$(CAMCOL)-$(FIELD4).fits $(URLBASE)/photoObj/301/$(RUN_STRIPPED)/$(CAMCOL)/photoObj-$(RUN6)-$(CAMCOL)-$(FIELD4).fits

$(SUBDIR2)/photoField-$(RUN6)-$(CAMCOL).fits :
	curl -L --create-dirs -o $(SUBDIR2)/photoField-$(RUN6)-$(CAMCOL).fits $(URLBASE)/photoObj/301/$(RUN_STRIPPED)/photoField-$(RUN6)-$(CAMCOL).fits

.PRECIOUS: $(SUBDIR3)/frame-%-$(RUN6)-$(CAMCOL)-$(FIELD4).fits.bz2 $(SUBDIR3)/fpM-$(RUN6)-%$(CAMCOL)-$(FIELD4).fit.gz

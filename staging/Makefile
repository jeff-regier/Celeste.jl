RUN := 3900
CAMCOL := 6
FIELD := 269

# strip leading zeros
RUN_STRIPPED := $(shell echo $(RUN) | sed 's/^0*//')
FIELD_STRIPPED := $(shell echo $(FIELD) | sed 's/^0*//')

RUN6 := $(shell printf '%06d' $(RUN_STRIPPED))
FIELD4 := $(shell printf '%04d' $(FIELD_STRIPPED))

STAGEDIR := $(CELESTE_STAGE_DIR)
SUBDIR2 := $(STAGEDIR)/$(RUN_STRIPPED)/$(CAMCOL)
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
      $(SUBDIR3)/photoObj-$(RUN6)-$(CAMCOL)-$(FIELD4).fits \
      $(SUBDIR2)/photoField-$(RUN6)-$(CAMCOL).fits \
      $(STAGEDIR)/field_extents.fits


$(STAGEDIR)/field_extents.fits :
	mkdir -p $(STAGEDIR)
	cp $(FIELD_EXTENTS) $(STAGEDIR)/field_extents.fits

$(SUBDIR3)/fpM-$(RUN6)-%$(CAMCOL)-$(FIELD4).fit :
	mkdir -p $(SUBDIR3)
	if [ -a $(SDSS_ROOT_DIR)/photo/redux/301/$(RUN_STRIPPED)/objcs/$(CAMCOL)/fpM-$(RUN6)-$*$(CAMCOL)-$(FIELD4).fit.gz ] ; \
	then \
		gunzip --stdout $(SDSS_ROOT_DIR)/photo/redux/301/$(RUN_STRIPPED)/objcs/$(CAMCOL)/fpM-$(RUN6)-$*$(CAMCOL)-$(FIELD4).fit.gz > $(SUBDIR3)/fpM-$(RUN6)-$*$(CAMCOL)-$(FIELD4).fit ; \
	fi;

$(SUBDIR3)/psField-$(RUN6)-$(CAMCOL)-$(FIELD4).fit :
	mkdir -p $(SUBDIR3)
	cp $(SDSS_ROOT_DIR)/photo/redux/301/$(RUN_STRIPPED)/objcs/$(CAMCOL)/psField-$(RUN6)-$(CAMCOL)-$(FIELD4).fit $(SUBDIR3)/psField-$(RUN6)-$(CAMCOL)-$(FIELD4).fit

$(SUBDIR3)/frame-%-$(RUN6)-$(CAMCOL)-$(FIELD4).fits :
	mkdir -p $(SUBDIR3)
	if [ -a $(SDSS_ROOT_DIR)/photoObj/frames/301/$(RUN_STRIPPED)/$(CAMCOL)/frame-$*-$(RUN6)-$(CAMCOL)-$(FIELD4).fits.bz2 ] ; \
	then \
		bzcat --keep $(SDSS_ROOT_DIR)/photoObj/frames/301/$(RUN_STRIPPED)/$(CAMCOL)/frame-$*-$(RUN6)-$(CAMCOL)-$(FIELD4).fits.bz2 > $(SUBDIR3)/frame-$*-$(RUN6)-$(CAMCOL)-$(FIELD4).fits ; \
	fi;

$(SUBDIR3)/photoObj-$(RUN6)-$(CAMCOL)-$(FIELD4).fits :
	mkdir -p $(SUBDIR3)
	cp $(SDSS_ROOT_DIR)/photoObj/301/$(RUN_STRIPPED)/$(CAMCOL)/photoObj-$(RUN6)-$(CAMCOL)-$(FIELD4).fits $(SUBDIR3)/photoObj-$(RUN6)-$(CAMCOL)-$(FIELD4).fits

$(SUBDIR2)/photoField-$(RUN6)-$(CAMCOL).fits :
	mkdir -p $(SUBDIR2)
	cp $(SDSS_ROOT_DIR)/photoObj/301/$(RUN_STRIPPED)/photoField-$(RUN6)-$(CAMCOL).fits $(SUBDIR2)/photoField-$(RUN6)-$(CAMCOL).fits 

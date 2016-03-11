URLBASE := http://data.sdss3.org/sas/dr12/boss
RUN := 3900
CAMCOL := 6
FIELD := 269

# strip leading zeros
RUN_STRIPPED := $(shell echo $(RUN) | sed 's/^0*//')
FIELD_STRIPPED := $(shell echo $(FIELD) | sed 's/^0*//')

RUN6 := $(shell printf '%06d' $(RUN_STRIPPED))
FIELD4 := $(shell printf '%04d' $(FIELD_STRIPPED))

all : fpM-$(RUN6)-u$(CAMCOL)-$(FIELD4).fit \
      fpM-$(RUN6)-g$(CAMCOL)-$(FIELD4).fit \
      fpM-$(RUN6)-r$(CAMCOL)-$(FIELD4).fit \
      fpM-$(RUN6)-i$(CAMCOL)-$(FIELD4).fit \
      fpM-$(RUN6)-z$(CAMCOL)-$(FIELD4).fit \
      frame-u-$(RUN6)-$(CAMCOL)-$(FIELD4).fits \
      frame-g-$(RUN6)-$(CAMCOL)-$(FIELD4).fits \
      frame-r-$(RUN6)-$(CAMCOL)-$(FIELD4).fits \
      frame-i-$(RUN6)-$(CAMCOL)-$(FIELD4).fits \
      frame-z-$(RUN6)-$(CAMCOL)-$(FIELD4).fits \
      psField-$(RUN6)-$(CAMCOL)-$(FIELD4).fit \
      photoObj-$(RUN6)-$(CAMCOL)-$(FIELD4).fits \
      photoField-$(RUN6)-$(CAMCOL).fits

fpM-$(RUN6)-%$(CAMCOL)-$(FIELD4).fit :
	wget --quiet $(URLBASE)/photo/redux/301/$(RUN_STRIPPED)/objcs/$(CAMCOL)/fpM-$(RUN6)-$*$(CAMCOL)-$(FIELD4).fit.gz
	gunzip fpM-$(RUN6)-$*$(CAMCOL)-$(FIELD4).fit.gz

psField-$(RUN6)-$(CAMCOL)-$(FIELD4).fit :
	wget --quiet $(URLBASE)/photo/redux/301/$(RUN_STRIPPED)/objcs/$(CAMCOL)/psField-$(RUN6)-$(CAMCOL)-$(FIELD4).fit

frame-%-$(RUN6)-$(CAMCOL)-$(FIELD4).fits :
	wget --quiet $(URLBASE)/photoObj/frames/301/$(RUN_STRIPPED)/$(CAMCOL)/frame-$*-$(RUN6)-$(CAMCOL)-$(FIELD4).fits.bz2
	bunzip2 frame-$*-$(RUN6)-$(CAMCOL)-$(FIELD4).fits.bz2

photoObj-$(RUN6)-$(CAMCOL)-$(FIELD4).fits :
	wget --quiet $(URLBASE)/photoObj/301/$(RUN_STRIPPED)/$(CAMCOL)/photoObj-$(RUN6)-$(CAMCOL)-$(FIELD4).fits

photoField-$(RUN6)-$(CAMCOL).fits :
	wget --quiet $(URLBASE)/photoObj/301/$(RUN_STRIPPED)/photoField-$(RUN6)-$(CAMCOL).fits

URLBASE := http://data.sdss3.org/sas/dr12/boss
RUN_SHORT := 4263
RUN := $(shell printf '%06d' $(RUN_SHORT))
CAMCOL := 5
FIELD := 0117

all : fpM-$(RUN)-u$(CAMCOL)-$(FIELD).fit \
      fpM-$(RUN)-g$(CAMCOL)-$(FIELD).fit \
      fpM-$(RUN)-r$(CAMCOL)-$(FIELD).fit \
      fpM-$(RUN)-i$(CAMCOL)-$(FIELD).fit \
      fpM-$(RUN)-z$(CAMCOL)-$(FIELD).fit \
      frame-u-$(RUN)-$(CAMCOL)-$(FIELD).fits \
      frame-g-$(RUN)-$(CAMCOL)-$(FIELD).fits \
      frame-r-$(RUN)-$(CAMCOL)-$(FIELD).fits \
      frame-i-$(RUN)-$(CAMCOL)-$(FIELD).fits \
      frame-z-$(RUN)-$(CAMCOL)-$(FIELD).fits \
      psField-$(RUN)-$(CAMCOL)-$(FIELD).fit \
      photoObj-$(RUN)-$(CAMCOL)-$(FIELD).fits \
      photoField-$(RUN)-$(CAMCOL).fits

fpM-$(RUN)-%$(CAMCOL)-$(FIELD).fit :
	wget --quiet $(URLBASE)/photo/redux/301/$(RUN_SHORT)/objcs/$(CAMCOL)/fpM-$(RUN)-$*$(CAMCOL)-$(FIELD).fit.gz
	gunzip fpM-$(RUN)-$*$(CAMCOL)-$(FIELD).fit.gz

psField-$(RUN)-$(CAMCOL)-$(FIELD).fit :
	wget --quiet $(URLBASE)/photo/redux/301/$(RUN_SHORT)/objcs/$(CAMCOL)/psField-$(RUN)-$(CAMCOL)-$(FIELD).fit

frame-%-$(RUN)-$(CAMCOL)-$(FIELD).fits :
	wget --quiet $(URLBASE)/photoObj/frames/301/$(RUN_SHORT)/$(CAMCOL)/frame-$*-$(RUN)-$(CAMCOL)-$(FIELD).fits.bz2
	bunzip2 frame-$*-$(RUN)-$(CAMCOL)-$(FIELD).fits.bz2

photoObj-$(RUN)-$(CAMCOL)-$(FIELD).fits :
	wget --quiet $(URLBASE)/photoObj/301/$(RUN_SHORT)/$(CAMCOL)/photoObj-$(RUN)-$(CAMCOL)-$(FIELD).fits

photoField-$(RUN)-$(CAMCOL).fits :
	wget --quiet $(URLBASE)/photoObj/301/$(RUN_SHORT)/photoField-$(RUN)-$(CAMCOL).fits

##
# Presentations
#
# @file
# @version 0.1

TEXS = $(wildcard src/*.md)
SLIDES = $(patsubst src/%.md, slides/slides_%.pdf, $(TEXS))
HANDOUTS = $(patsubst src/%.md, handouts/handout_%.pdf, $(TEXS))

all: $(SLIDES)
handouts: $(HANDOUTS)
slides: $(SLIDES)

slides/slides_%.pdf: src/%.md
	cd src && \
	pandoc -s --dpi=300 --slide-level 2 --toc --listings --shift-heading-level=0 -V classoption:aspectratio=169 -t beamer $*.md -o ../slides/slides_$*.pdf

handouts/handout_%.pdf: src/%.md
	cd src && \
	pandoc -s --dpi=300 --slide-level 2 --toc --listings --shift-heading-level=0 -V handout -V classoption:aspectratio=169 -t beamer $*.md -o ../handouts/handout_$*.pdf


# end

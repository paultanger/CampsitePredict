#!/bin/sh
git add index.ipynb
git commit -m "update index.ipynb"
jupyter nbconvert --to slides index.ipynb
git checkout gh-pages
mv index.slides.html index.html
git add index.html
git commit -m "update index.html"
git push
git checkout master
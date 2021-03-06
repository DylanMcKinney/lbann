## LBANN Documentation via Doxygen
### Updating the Github Pages
This documentation is generated and deployed to github pages automatically using Bamboo CI. To trigger an update to the pages add "gen_doxygen" to your commit message.
```
git commit -m "Added example.cpp gen_doxygen"
```

### Adding to the documentation
**Adding to Doxygen (general):** For general doxygen formatting/syntax see [Doxygen User Manuel](http://www.stack.nl/~dimitri/doxygen/manual/index.html).

**Adding pages:** To add a page to the documentation index create a .dox file and add it to the INPUT field found in Doxyfile.in. These pages appear in the index in the order listed in the INPUT field. Examples of how to link to this page from other pages can be found in mainpage.dox 

**Adding to pages**: Our documentation uses pages to give users extended details on a variety of LBANN components without having to dig into the implementation details. To add a new component class to the associated page create a subsection and use the \copydetails doxygen command

```
\copydetails lbann::example_layer_class
```

To link the extended implementation details just need to link the full class name, i.e `lbann::example_layer_class`. This will link to the class page doxygen automatically generates.

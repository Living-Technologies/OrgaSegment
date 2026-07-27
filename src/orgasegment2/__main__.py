#!/usr/bin/env python3

import orgasegment2

from optparse import OptionParser



if __name__=="__main__":
    print(orgasegment2.version)
    parser = OptionParser()
    parser.add_option("-d", "--directory", dest="directory",
                  help="Folder containing input images", metavar="DIRECTORY")
    parser.add_option("-m", "--model", dest="modelfile", help="cellpose model for inference", metavar="FILE")
    parser.add_option("-p", "--predict", default=False, action="store_true", dest="predict", help="Include inference", metavar="BOOLEAN")
    parser.add_option("-t", "--track", default=False, action="store_true", dest="track", help="Include tracking", metavar="BOOLEAN")
    parser.add_option("-o", "--output", dest="output",
                  help="Folder to save predictions, previews and datafiles", metavar="DIRECTORY", default="output")
    parser.add_option( "-r", "--regex", 
                       help="Regular Expression for capturing WELL and T values", 
                       default='.*(?P<WELL>[A-Z]{1}[0-9]{1,2}).*[tT](?P<T>[0-9]{1,2}).*',
                       dest="regex")
    parser.add_option( "-s", "--search_range", 
                       help="The tracking distance", 
                       default="50",
                       dest="search_range")
    parser.add_option( "-y", "--tracking_memory", 
                       help="Trackpy parameter for linking tracks", 
                       default="0",
                       dest="tracking_memory")
    (options, args) = parser.parse_args()
    
    if options.predict:
        orgasegment2.predict( options.modelfile, options.directory, options.output )
    if options.track:
        #regex = '.*(?P<WELL>[A-Z]{1}[0-9]{1,2}).*t(?P<T>[0-9]{1,2}).*'
        orgasegment2.track( options.output, options.regex, options.search_range, options.tracking_memory )
    

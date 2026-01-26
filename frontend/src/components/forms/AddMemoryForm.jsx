import React, { useState } from 'react';
import { Plus, Upload, X, MapPin, Calendar, Users, Loader2 } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog';
import { Badge } from '@/components/ui/badge';
import { memoriesApi, uploadApi } from '@/services/api';
import { toast } from 'sonner';

const categoryOptions = [
  { value: 'milestone', label: 'Milestone (Wedding, Birth, Graduation)' },
  { value: 'family', label: 'Family Moment' },
  { value: 'travel', label: 'Travel / Vacation' },
  { value: 'celebration', label: 'Celebration / Holiday' },
  { value: 'other', label: 'Other' },
];

export const AddMemoryForm = ({ familyMembers = [], onSuccess, onClose }) => {
  const [isOpen, setIsOpen] = useState(false);
  const [loading, setLoading] = useState(false);
  const [uploadingPhoto, setUploadingPhoto] = useState(false);
  const [personInput, setPersonInput] = useState('');
  
  const [formData, setFormData] = useState({
    title: '',
    date: '',
    year: new Date().getFullYear(),
    location: '',
    description: '',
    people: [],
    photos: [],
    category: '',
  });

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({ ...prev, [name]: value }));
  };

  const handleYearChange = (e) => {
    const value = parseInt(e.target.value) || new Date().getFullYear();
    setFormData(prev => ({ ...prev, year: value }));
  };

  const handleCategoryChange = (value) => {
    setFormData(prev => ({ ...prev, category: value }));
  };

  const handlePhotoUpload = async (e) => {
    const files = Array.from(e.target.files);
    if (files.length === 0) return;
    
    setUploadingPhoto(true);
    try {
      const urls = await Promise.all(files.map(file => uploadApi.uploadFile(file)));
      setFormData(prev => ({ ...prev, photos: [...prev.photos, ...urls] }));
      toast.success('Photo uploaded successfully');
    } catch (error) {
      toast.error('Failed to upload photo');
    } finally {
      setUploadingPhoto(false);
    }
  };

  const removePhoto = (index) => {
    setFormData(prev => ({
      ...prev,
      photos: prev.photos.filter((_, i) => i !== index)
    }));
  };

  const addPerson = (name) => {
    if (name && !formData.people.includes(name)) {
      setFormData(prev => ({ ...prev, people: [...prev.people, name] }));
    }
    setPersonInput('');
  };

  const removePerson = (name) => {
    setFormData(prev => ({
      ...prev,
      people: prev.people.filter(p => p !== name)
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!formData.title || !formData.date || !formData.category || !formData.description) {
      toast.error('Please fill in the required fields');
      return;
    }
    
    setLoading(true);
    try {
      await memoriesApi.create(formData);
      toast.success('Memory saved successfully!');
      setFormData({
        title: '',
        date: '',
        year: new Date().getFullYear(),
        location: '',
        description: '',
        people: [],
        photos: [],
        category: '',
      });
      setIsOpen(false);
      if (onSuccess) onSuccess();
      if (onClose) onClose();
    } catch (error) {
      toast.error('Failed to save memory');
    } finally {
      setLoading(false);
    }
  };

  const dialogContent = (
    <form onSubmit={handleSubmit} className="space-y-6 mt-4">
      {/* Title */}
      <div>
        <Label htmlFor="title" className="text-lg font-semibold">Memory Title *</Label>
        <Input
          id="title"
          name="title"
          value={formData.title}
          onChange={handleInputChange}
          placeholder="e.g., Our Wedding Day, Family Vacation to Paris"
          className="mt-2 h-14 text-lg"
          required
        />
      </div>

      {/* Date and Year */}
      <div className="grid grid-cols-2 gap-4">
        <div>
          <Label htmlFor="date" className="text-lg font-semibold flex items-center gap-2">
            <Calendar className="h-5 w-5" /> Date *
          </Label>
          <Input
            id="date"
            name="date"
            value={formData.date}
            onChange={handleInputChange}
            placeholder="e.g., June 15, 1972"
            className="mt-2 h-14 text-lg"
            required
          />
        </div>
        <div>
          <Label htmlFor="year" className="text-lg font-semibold">Year *</Label>
          <Input
            id="year"
            name="year"
            type="number"
            value={formData.year}
            onChange={handleYearChange}
            min="1900"
            max={new Date().getFullYear()}
            className="mt-2 h-14 text-lg"
            required
          />
        </div>
      </div>

      {/* Category */}
      <div>
        <Label className="text-lg font-semibold">Category *</Label>
        <Select value={formData.category} onValueChange={handleCategoryChange}>
          <SelectTrigger className="mt-2 h-14 text-lg">
            <SelectValue placeholder="Select memory type" />
          </SelectTrigger>
          <SelectContent>
            {categoryOptions.map(opt => (
              <SelectItem key={opt.value} value={opt.value} className="text-lg py-3">
                {opt.label}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>

      {/* Location */}
      <div>
        <Label htmlFor="location" className="text-lg font-semibold flex items-center gap-2">
          <MapPin className="h-5 w-5" /> Location
        </Label>
        <Input
          id="location"
          name="location"
          value={formData.location}
          onChange={handleInputChange}
          placeholder="e.g., St. Mary's Church, Paris, France"
          className="mt-2 h-14 text-lg"
        />
      </div>

      {/* People */}
      <div>
        <Label className="text-lg font-semibold flex items-center gap-2">
          <Users className="h-5 w-5" /> People in this Memory
        </Label>
        <div className="mt-2 flex gap-2">
          <Input
            value={personInput}
            onChange={(e) => setPersonInput(e.target.value)}
            placeholder="Add a person's name"
            className="h-14 text-lg flex-1"
            onKeyPress={(e) => {
              if (e.key === 'Enter') {
                e.preventDefault();
                addPerson(personInput);
              }
            }}
          />
          <Button
            type="button"
            variant="outline"
            className="h-14 px-6"
            onClick={() => addPerson(personInput)}
          >
            Add
          </Button>
        </div>
        
        {/* Quick add from family members */}
        {familyMembers.length > 0 && (
          <div className="mt-3">
            <p className="text-sm text-muted-foreground mb-2">Quick add from family:</p>
            <div className="flex flex-wrap gap-2">
              {familyMembers.slice(0, 6).map(member => (
                <Button
                  key={member.id}
                  type="button"
                  variant="ghost"
                  size="sm"
                  className={`${formData.people.includes(member.name) ? 'bg-primary/20' : ''}`}
                  onClick={() => {
                    if (formData.people.includes(member.name)) {
                      removePerson(member.name);
                    } else {
                      addPerson(member.name);
                    }
                  }}
                >
                  {member.name}
                </Button>
              ))}
            </div>
          </div>
        )}
        
        {formData.people.length > 0 && (
          <div className="flex flex-wrap gap-2 mt-3">
            {formData.people.map(person => (
              <Badge
                key={person}
                variant="secondary"
                className="text-base py-1 px-3 gap-2"
              >
                {person}
                <button
                  type="button"
                  onClick={() => removePerson(person)}
                  className="hover:text-destructive"
                >
                  <X className="h-4 w-4" />
                </button>
              </Badge>
            ))}
          </div>
        )}
      </div>

      {/* Description */}
      <div>
        <Label htmlFor="description" className="text-lg font-semibold">Tell the Story *</Label>
        <Textarea
          id="description"
          name="description"
          value={formData.description}
          onChange={handleInputChange}
          placeholder="Describe this memory in detail. What happened? How did you feel? What made it special?"
          className="mt-2 min-h-[150px] text-lg"
          required
        />
      </div>

      {/* Photos */}
      <div>
        <Label className="text-lg font-semibold flex items-center gap-2">
          <Upload className="h-5 w-5" /> Photos
        </Label>
        <div className="mt-2">
          <input
            type="file"
            accept="image/*"
            multiple
            onChange={handlePhotoUpload}
            className="hidden"
            id="memory-photo-upload"
          />
          <label htmlFor="memory-photo-upload">
            <Button
              type="button"
              variant="outline"
              className="w-full h-14 text-lg cursor-pointer"
              disabled={uploadingPhoto}
              asChild
            >
              <span>
                {uploadingPhoto ? (
                  <><Loader2 className="h-5 w-5 mr-2 animate-spin" /> Uploading...</>
                ) : (
                  <><Upload className="h-5 w-5 mr-2" /> Upload Photos</>
                )}
              </span>
            </Button>
          </label>
        </div>
        {formData.photos.length > 0 && (
          <div className="flex flex-wrap gap-2 mt-3">
            {formData.photos.map((url, index) => (
              <div key={index} className="relative">
                <img src={url} alt="" className="h-20 w-20 object-cover rounded-lg" />
                <button
                  type="button"
                  onClick={() => removePhoto(index)}
                  className="absolute -top-2 -right-2 bg-destructive text-destructive-foreground rounded-full p-1"
                >
                  <X className="h-4 w-4" />
                </button>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Submit */}
      <Button
        type="submit"
        variant="accessible"
        className="w-full"
        disabled={loading}
      >
        {loading ? (
          <><Loader2 className="h-5 w-5 mr-2 animate-spin" /> Saving...</>
        ) : (
          <><Plus className="h-5 w-5 mr-2" /> Save Memory</>
        )}
      </Button>
    </form>
  );

  // If used as standalone (with onClose prop), render just the form
  if (onClose) {
    return dialogContent;
  }

  // Otherwise render with dialog trigger
  return (
    <Dialog open={isOpen} onOpenChange={setIsOpen}>
      <DialogTrigger asChild>
        <Button variant="accessible" className="gap-2">
          <Plus className="h-5 w-5" />
          Add Memory
        </Button>
      </DialogTrigger>
      <DialogContent className="max-w-lg max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle className="text-2xl font-display flex items-center gap-2">
            <Calendar className="h-6 w-6 text-primary" />
            Add New Memory
          </DialogTitle>
        </DialogHeader>
        {dialogContent}
      </DialogContent>
    </Dialog>
  );
};

export default AddMemoryForm;
